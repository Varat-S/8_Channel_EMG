"""Run a compact multi-posture model in the local 3D hand interface."""
from collections import deque
from http import HTTPStatus
from http.server import ThreadingHTTPServer
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
import webapp
from emg.compact_model import CompactFilter
from emg.dataset_v2 import V2_COMPACT_GESTURES
from emg.features import extract_time_domain_features
from emg.model import load_model
from emg.preprocessing import apply_spatial_reference
from emg.sources import SerialFrameSource


FIVE_CORE_GESTURES = ("rest", "fist", "splayed", "wrist_up", "wrist_down")
FOUR_CORE_GESTURES = ("rest", "fist", "splayed", "wrist_up")
SUPPORTED_CLASS_SETS = {
    frozenset(FOUR_CORE_GESTURES),
    frozenset(FIVE_CORE_GESTURES),
    frozenset(V2_COMPACT_GESTURES),
}


class CompactController(webapp.PredictionController):
    def __init__(self,args,hub):
        artifact=load_model(args.model)
        labels=frozenset(str(label) for label in artifact['model'].classes_)
        if labels not in SUPPORTED_CLASS_SETS:
            raise ValueError('expected the compact four-, five- or seven-class model')
        rejection=artifact.get('rejection_profile') or {}
        self.compact_labels=labels
        self.output_threshold=float(rejection.get('minimum_output_confidence',0.0))
        self.output_policy=str(rejection.get('policy','hold'))
        self.model_description=str(artifact.get('model_description') or '')
        if not 0<=self.output_threshold<=1 or self.output_policy not in {'hold','rest'}:
            raise ValueError('invalid compact rejection profile')
        self._last_accepted_output='rest'
        super().__init__(args,hub)

    def _expected_labels(self):
        return set(self.compact_labels)

    def config_event(self):
        result=super().config_event()
        count=len(self.compact_labels)
        rejection=(f' Outputs below {self.output_threshold:.0%} confidence hold the previous pose.'
                   if self.output_threshold else '')
        result.update(calibrationGestures=[g for g in V2_COMPACT_GESTURES if g!='rest' and g in self.compact_labels],
                      freshCalibrationEnabled=False,
                      outputConfidenceThreshold=self.output_threshold,
                      outputRejectionPolicy=self.output_policy,
                      modelDescription=(self.model_description or f'{count}-class compact EMG model.')
                      + rejection + ' Keep the electrodes in the recorded positions.')
        return result

    def start(self):
        self._last_accepted_output='rest'
        return super().start()

    def reset(self):
        self._last_accepted_output='rest'
        return super().reset()

    def disarm(self):
        self._last_accepted_output='rest'
        super().disarm()

    def _publish_prediction(self,gesture,confidence):
        now=time.monotonic()
        stable=self.detector.update(gesture,confidence,now)
        just_armed=self.gate.accept(stable,now) if stable is not None else False
        rejected=self.gate.armed and confidence<self.output_threshold
        if not self.gate.armed:
            output='rest'
        elif rejected:
            output=self._last_accepted_output if self.output_policy=='hold' else 'rest'
        else:
            output=gesture
            self._last_accepted_output=gesture
        self.hub.publish({'type':'prediction','rawGesture':gesture,'gesture':output,
                          'confidence':round(confidence,4),'rejected':rejected,
                          'outputConfidenceThreshold':self.output_threshold,
                          'armed':self.gate.armed,'justArmed':just_armed,
                          'progress':self.gate.progress,'expected':self.gate.expected,
                          'wakeSequence':list(self.gate.sequence)})

    def start_calibration(self):
        self.hub.publish({'type':'error','message':'This page uses the multi-posture model. Collect new chunks to retrain it.'})
        return False

    def _new_filter(self,artifact):
        profile=artifact['compact_profile']
        return CompactFilter(profile['notch_hz'],profile['notch_q'])

    def _collect_baseline(self,source,artifact,signal_filter,spatial_reference,sample_rate_hz):
        profile=artifact['compact_profile']
        if profile['normalization']=='fixed_reference':
            scale=np.asarray(profile['fixed_reference_scale'],dtype=float)
            if scale.shape!=(8,) or not np.all(np.isfinite(scale)) or np.any(scale<=1e-8):
                raise ValueError('invalid model reference scale')
            return scale,signal_filter
        return super()._collect_baseline(source,artifact,signal_filter,spatial_reference,sample_rate_hz)

    def _serial_worker(self):
        try:
            artifact=load_model(self.active_model_path)
            model=artifact['model']
            if frozenset(model.classes_)!=self.compact_labels:
                raise ValueError('model labels changed after controller initialization')
            profile=artifact['compact_profile']
            if profile['version']!=1 or float(artifact['sample_rate_hz'])!=125:
                raise ValueError('incompatible compact processing profile')
            signal_filter=self._new_filter(artifact)
            self.hub.publish({'type':'status','phase':'connecting','message':f'Opening {self.args.port} for the multi-posture model'})
            with SerialFrameSource(port=self.args.port,baud=self.args.baud) as source:
                scale,signal_filter=self._collect_baseline(source,artifact,signal_filter,'common_average',125)
                buffer=deque(maxlen=int(artifact['window_size']))
                history=deque(maxlen=int(profile['smoothing_windows']))
                since_window=0;guard=int(profile['filter_reset_guard_samples']);retries=0
                available=', '.join(sorted(self.compact_labels-{'rest'}))
                self.hub.publish({'type':'status','phase':'live','message':f'Live model ready. Wake with SPLAYED → FIST → SPLAYED. Available active poses: {available}.'})
                while not self.stop_event.is_set():
                    try:
                        sample=source.read_sample();retries=0
                    except TimeoutError:
                        retries+=1
                        if retries>3:
                            raise TimeoutError('receiver did not resume after three reconnect attempts')
                        self.hub.publish({'type':'status','phase':'reconnecting','message':f'Receiver paused; retrying {retries}/3'})
                        source.reconnect();signal_filter=self._new_filter(artifact)
                        buffer.clear();history.clear();since_window=0
                        guard=int(profile['filter_reset_guard_samples'])
                        self.gate.disarm();self.detector.reset();self._last_accepted_output='rest'
                        continue
                    processed=signal_filter.update(apply_spatial_reference(sample.values,'common_average'))/scale
                    if guard:
                        guard-=1;continue
                    buffer.append(processed)
                    if len(buffer)<buffer.maxlen:
                        continue
                    due=since_window%int(artifact['hop_size'])==0;since_window+=1
                    if not due:
                        continue
                    features=extract_time_domain_features(np.asarray(buffer),artifact['thresholds'])
                    history.append(model.predict_proba(features[None,:])[0])
                    probabilities=np.mean(history,axis=0);best=int(np.argmax(probabilities))
                    self._publish_prediction(str(model.classes_[best]),float(probabilities[best]))
        except Exception as error:
            self.hub.publish({'type':'error','message':f'{type(error).__name__}: {error}'})


def make_handler(controller,hub):
    base=webapp.make_handler(controller,hub)
    class Handler(base):
        def do_GET(self):
            if self.path=='/api/state':
                self._json({'running':controller.running,'model':str(controller.active_model_path),'event':hub.last_event})
            else:
                super().do_GET()
    return Handler


def main():
    args=webapp.parse_args()
    artifact=load_model(args.model)
    if ('compact_profile' not in artifact
            or frozenset(artifact['model'].classes_) not in SUPPORTED_CLASS_SETS):
        raise ValueError('Select a supported compact four-, five- or seven-class model')
    hub=webapp.EventHub();controller=CompactController(args,hub)
    server=ThreadingHTTPServer((args.host,args.http_port),make_handler(controller,hub))
    url=f'http://{args.host}:{args.http_port}'
    print(f'Compact {len(artifact["model"].classes_)}-class model: {args.model}',flush=True)
    print(f'Local hand display: {url}',flush=True)
    if not args.no_browser:
        threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop();server.server_close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
