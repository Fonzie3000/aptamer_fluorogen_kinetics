"""
Trace-event detection and feature extraction.

Both helpers operate on single-molecule fluorescence intensity traces that have
already been segmented by a two-state Gaussian HMM (state 0 = background,
state 1 = bound). The HMM itself is fit in ``core.hmm_engine`` — nothing here
fits a model.

  • ``EventExtractor.extract_events`` walks a per-frame state sequence and emits
    one dict per binding event (start/end time and index, duration, mean
    intensity, per-event SNR).
  • ``EventFeatureExtractor.batch_extract_features`` turns those events into the numeric
    (N, 11) feature matrix consumed by the deterministic keep/reject criteria
    in the Trace Filter step.

"""

import numpy as np
import numba

from .data_types import BindingEvent


def _event_snr(mean_intensity, bg_mean, bg_std):
    """Per-event signal-to-noise ratio: ``(mean_intensity - bg_mean) / bg_std``.

    ``bg_std`` is floored at 1e-6 to avoid divide-by-zero on flat traces. Used by
    :meth:`EventExtractor.extract_events`, where the background statistics come
    from the HMM background state (state 0). :meth:`EventFeatureExtractor.batch_extract_features`
    recomputes its own SNR feature over a background mask that excludes *all*
    events, so the two SNR values are close but not identical.
    """
    return (mean_intensity - bg_mean) / max(bg_std, 1e-6)


class EventExtractor:
    """Parses a two-state HMM state sequence into discrete binding events.

    Despite the name, this class does **not** fit an HMM — the fitting lives in
    ``core.hmm_engine``. It consumes the resulting per-frame state sequence and
    segments it into events with a simple rising/falling-edge scan.
    """

    def __init__(self, n_components=2):
        """
        Args:
            n_components: State count passed to the HMM worker
                (2: background + bound). The event scan identifies state 1.
        """
        self.n_components = n_components

    def extract_events(self, time_axis, states, data, min_snr=0.0):
        """Segment a binary state sequence into a list of binding events.

        Args:
            time_axis (np.ndarray): Per-frame time (or frame index) values.
            states (np.ndarray): Per-frame HMM states (0 = background, 1 = bound).
            data (np.ndarray): Raw intensity trace (same length as ``states``).
            min_snr (float): Drop events whose SNR = (event_mean - bg_mean)/bg_std
                falls below this. 0.0 (default) disables the filter.

        Returns:
            list[BindingEvent]: one per event, with ``Start, End, Duration,
            MeanIntensity, StartIdx, EndIdx, SNR, StartTrunc, EndTrunc``.
            ``End`` and ``EndIdx`` are exclusive bounds.
            Background statistics for SNR are taken from the state-0 frames.

            ``StartTrunc``/``EndTrunc`` flag an event that was already in progress at
            the first frame or still in progress at the last, so its true dwell time
            is unknown. EVERY detected event is returned regardless — the flag is
            consumed by ``core.kinetics`` when fitting rates, not here.
        """
        # Pre-compute background stats for SNR filtering
        bg_mask = (states == 0)
        bg_data = data[bg_mask] if np.any(bg_mask) else data
        bg_mean = np.mean(bg_data)
        bg_std = np.std(bg_data) if len(bg_data) > 1 else 1.0

        events = []
        is_event = False
        start_idx = 0

        # Simple finite state machine loop
        for i, state in enumerate(states):
            if state == 1 and not is_event:
                # Rising edge
                is_event = True
                start_idx = i
            elif state != 1 and is_event:
                # Falling edge
                is_event = False
                end_idx = i

                # Calculate specs
                duration = time_axis[end_idx] - time_axis[start_idx]
                event_data = data[start_idx:end_idx]
                mean_intensity = np.mean(event_data)

                # Compute per-event SNR
                event_snr = _event_snr(mean_intensity, bg_mean, bg_std)

                # Apply SNR filter
                if min_snr > 0 and event_snr < min_snr:
                    continue  # Skip this event — below noise floor

                events.append(BindingEvent(
                    Start=float(time_axis[start_idx]),
                    End=float(time_axis[end_idx]),
                    Duration=float(duration),
                    MeanIntensity=round(mean_intensity, 2),
                    StartIdx=int(start_idx),
                    EndIdx=int(end_idx),
                    SNR=round(event_snr, 2),
                    # Already bound when acquisition started -> true dwell unknown.
                    StartTrunc=int(start_idx == 0),
                    EndTrunc=0,
                ))

        # Handle case where the trace ends while still in an event
        if is_event:
            end_idx = len(states)
            # There is no frame past the end to read the closing time from, so step
            # one sample beyond the last one. Without this the event comes out a
            # frame SHORT and its End is inclusive while every other event's is
            # exclusive: the falling-edge branch above uses time_axis[end_idx], which
            # for a run ending at the final frame would be time_axis[len(states)].
            # On the default frame-index axis this reduces exactly to
            # duration = end_idx - start_idx and End = end_idx, matching that branch.
            dt = (time_axis[-1] - time_axis[-2]) if len(time_axis) >= 2 else 1.0
            duration = (time_axis[end_idx - 1] - time_axis[start_idx]) + dt
            event_data = data[start_idx:end_idx]
            mean_intensity = np.mean(event_data)

            # Compute per-event SNR
            event_snr = _event_snr(mean_intensity, bg_mean, bg_std)

            if min_snr <= 0 or event_snr >= min_snr:
                events.append(BindingEvent(
                    Start=float(time_axis[start_idx]),
                    End=float(time_axis[start_idx] + duration),
                    Duration=float(duration),
                    MeanIntensity=round(mean_intensity, 2),
                    StartIdx=int(start_idx),
                    EndIdx=int(end_idx),
                    SNR=round(event_snr, 2),
                    # This branch IS the end-truncated case: the trace ran out while
                    # the molecule was still bound, so only a lower bound on the
                    # dwell was observed.
                    StartTrunc=int(start_idx == 0),
                    EndTrunc=1,
                ))

        return events


@numba.jit(nopython=True, nogil=True, cache=True)
def _extract_features_numba(trace, starts, ends, durations, intensities,
                            bg_mean, bg_std, n_events, feat):
    """Fill the pre-allocated (N, 11) feature matrix ``feat`` in place.

    Columns: 0 duration, 1 mean intensity, 2 std-dev in event, 3 event/bg ratio,
    4 number of events, 5 rise slope, 6 fall slope, 7 symmetry,
    8 background recovery, 9 local SNR, 10 trace mean.
    """
    T = len(trace)
    pre_window = 3
    post_window = 3
    trace_mean = 0.0
    for i in range(T):
        trace_mean += trace[i]
    trace_mean /= T

    for i in range(len(starts)):
        s = starts[i]
        e_idx = ends[i]
        dur = durations[i]
        inten = intensities[i]

        # Col 0: duration
        feat[i, 0] = dur
        # Col 1: intensity
        feat[i, 1] = inten
        # Col 4: num events (same for all)
        feat[i, 4] = n_events
        # Col 10: trace mean
        feat[i, 10] = trace_mean

        # Col 3: ratio
        feat[i, 3] = inten / max(bg_mean, 1e-6)
        # Col 9: local SNR
        feat[i, 9] = (inten - bg_mean) / max(bg_std, 1e-6)

        # Col 2: StdDev in event
        end_clip = min(e_idx, T)
        slice_len = end_clip - s
        if slice_len > 1:
            mean_p = 0.0
            for j in range(s, end_clip):
                mean_p += trace[j]
            mean_p /= slice_len
            var_p = 0.0
            for j in range(s, end_clip):
                var_p += (trace[j] - mean_p) ** 2
            feat[i, 2] = (var_p / slice_len) ** 0.5
        else:
            feat[i, 2] = 0.0

        # Rise slope (col 5)
        pre_s = max(0, s - pre_window)
        if s > 0 and s > pre_s:
            pre_mean = 0.0
            for j in range(pre_s, s):
                pre_mean += trace[j]
            pre_mean /= (s - pre_s)
        else:
            pre_mean = bg_mean
        rise = (inten - pre_mean) / max(dur, 1.0)
        feat[i, 5] = rise

        # Fall slope (col 6)
        post_e = min(T, e_idx + post_window)
        if e_idx < T and post_e > e_idx:
            post_mean = 0.0
            for j in range(e_idx, post_e):
                post_mean += trace[j]
            post_mean /= (post_e - e_idx)
        else:
            post_mean = bg_mean
        fall = (inten - post_mean) / max(dur, 1.0)
        feat[i, 6] = fall

        # Symmetry (col 7)
        max_slope = max(rise, fall)
        if max_slope > 0:
            feat[i, 7] = min(rise, fall) / max_slope
        else:
            feat[i, 7] = 1.0

        # Background recovery (col 8)
        feat[i, 8] = 1.0 - abs(post_mean - pre_mean) / max(inten, 1e-6)


class EventFeatureExtractor:
    """Extracts the per-event numeric feature matrix used for trace filtering.

    This is the feature *extractor*, not the filter: the deterministic
    keep/reject filters themselves are applied in the GUI Trace Filter
    controller against the ``feature_names`` columns produced here.
    """

    def __init__(self):
        # Column order must match _extract_features_numba (and the filter
        # parameter names in gui.panels.TraceFilterPanel).
        self.feature_names = [
            "Event duration (frames)", "Event intensity (mean)", "StdDev in event",
            "Event/BG ratio", "Number of events", "Rise slope", "Fall slope",
            "Symmetry", "Background recovery", "Event SNR", "Trace mean"
        ]

    def batch_extract_features(self, events, trace_data):
        """Extract features for ALL events of a molecule at once.

        Returns an (N, 11) numpy array where N = len(events), columns ordered as
        :attr:`feature_names`. The numba inner loop does the per-event work; the
        background statistics use a mask that excludes every event's frames.
        """
        n = len(events)
        if n == 0:
            return np.empty((0, 11), dtype=np.float64)

        T = len(trace_data)
        trace = np.ascontiguousarray(trace_data, dtype=np.float64)

        # Pre-extract event bounds as arrays
        starts = np.array([(e.StartIdx if e.StartIdx >= 0 else int(e.Start)) for e in events], dtype=np.int64)
        ends = np.array([(e.EndIdx if e.EndIdx >= 0 else int(e.End)) for e in events], dtype=np.int64)
        durations = np.array([e.Duration for e in events], dtype=np.float64)
        intensities = np.array([e.MeanIntensity for e in events], dtype=np.float64)

        # Background mask: everything NOT in any event
        bg_mask = np.ones(T, dtype=np.bool_)
        for s, e in zip(starts, ends):
            bg_mask[s:min(e, T)] = False
        bg_data = trace[bg_mask]
        bg_mean = np.mean(bg_data) if len(bg_data) > 0 else 1.0
        bg_std = np.std(bg_data) if len(bg_data) > 1 else 1.0

        # Allocate feature matrix and fill via numba
        feat = np.empty((n, 11), dtype=np.float64)
        _extract_features_numba(
            trace, starts, ends, durations, intensities,
            bg_mean, bg_std, float(n), feat
        )
        return feat
