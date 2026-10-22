"""Select and train a same-application multi-posture seven-class decoder."""
from pathlib import Path
import argparse
import json
import sys
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))

import joblib
import numpy as np
from emg.compact_model import read_collection,make_dataset,compact_split
from emg.evaluation.benchmark import DEFAULT_CONFIG,evaluate_model,write_report
from emg.evaluation.dataset_v2 import score_v2
from emg.evaluation.metrics import calculate_metrics
from emg.model import _new_pipeline,save_model


def train(collection_path,output,model_path):
    output=Path(output);model_path=Path(model_path)
    if output.exists() or model_path.exists():
        raise ValueError('Use fresh output paths; existing reports/models are preserved.')
    collection,recordings=read_collection(collection_path)
    variants=[('multi_posture_session',None,'session_baseline'),
              ('multi_posture_fixed',None,'fixed_reference'),
              ('multi_posture_notch50_session',50.0,'session_baseline'),
              ('multi_posture_notch50_fixed',50.0,'fixed_reference')]
    choices=[];cache={}
    for name,notch,normalization in variants:
        dataset,conditions,scale=make_dataset(recordings,notch_hz=notch,normalization=normalization)
        cache[name]=(dataset,conditions,scale)
        metrics=[]
        for seed in range(7,12):
            train_indices,test_indices=compact_split(dataset,seed)
            model=_new_pipeline(7).fit(dataset.data.features[train_indices],dataset.data.labels[train_indices])
            result=evaluate_model(model,dataset,test_indices,DEFAULT_CONFIG)
            metrics.append(result.metrics['smoothed']['macro_f1'])
        choices.append({'variant':name,'notch_hz':notch,'normalization':normalization,
                        'validation_macro_f1_mean':float(np.mean(metrics)),
                        'validation_macro_f1_std':float(np.std(metrics,ddof=1)),
                        'validation_macro_f1_by_seed':metrics})
        print(json.dumps(choices[-1]),flush=True)
    # These are model-selection validation scores, not prospective test accuracy.
    selected=max(choices,key=lambda c:c['validation_macro_f1_mean'])
    dataset,conditions,scale=cache[selected['variant']]
    train_indices,test_indices=compact_split(dataset,7)
    evaluation_model=_new_pipeline(7).fit(dataset.data.features[train_indices],dataset.data.labels[train_indices])
    independent=np.array([i for i,w in enumerate(dataset.windows) if w.phase in ('rest_pre','rest_post')])
    result=score_v2(evaluation_model,dataset,test_indices,independent,DEFAULT_CONFIG)
    selected_train_ids=sorted(set(dataset.data.groups[train_indices]))
    selected_test_ids=sorted(set(dataset.data.groups[test_indices]))
    deployment_indices=np.array([i for i,w in enumerate(dataset.windows) if w.phase=='gesture'])
    deployment_ids=sorted(set(dataset.data.groups[deployment_indices]))
    config={**DEFAULT_CONFIG,'experiment':'compact_multi_posture_model_selection',
            'candidates':choices,'selected':selected,'seed':7,'choice_seeds':list(range(7,12)),
            'sources':[r.audit for r in recordings],
            'train_trial_ids':selected_train_ids,'validation_trial_ids':selected_test_ids,
            'deployment_training_trial_ids':deployment_ids,
            'evaluation_scope':'internal whole-trial validation on recorded positions; hyperparameters selected on these scores',
            'unfamiliar_position_test_used':False,'normalization_note':'Extended chunk uses confirmed posture-specific baseline; fixed-reference candidates use only the original reference baseline for all chunks.',
            'reconnect_handling':'reset causal filters at acquisition discontinuities; reject interrupted gesture holds; guard segment edges',
            'source_code_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                                  ['train_compact_model.py','src/emg/compact_model.py','src/emg/features.py','src/emg/model.py']}}
    write_report(output,result,dataset,config,evaluation_model)
    # Compare against reference-only training on exactly the same held-out trials.
    reference_id=recordings[0].metadata['session_id']
    reference_train=np.array([i for i in train_indices if dataset.windows[i].session_id==reference_id])
    reference_model=_new_pipeline(7).fit(dataset.data.features[reference_train],dataset.data.labels[reference_train])
    reference_result=evaluate_model(reference_model,dataset,test_indices,DEFAULT_CONFIG)
    (output/'reference_only_comparison.json').write_text(json.dumps(reference_result.metrics,indent=2)+'\n')
    by_condition={}
    for session in {w.session_id for w in dataset.windows}:
        selected_rows=np.array([w.session_id==session for w in (dataset.windows[i] for i in result.test_indices)])
        classification=np.array([dataset.windows[i].phase=='gesture' for i in result.test_indices])
        mask=selected_rows & classification
        by_condition[session]=calculate_metrics(dataset.data.labels[result.test_indices[mask]],result.smoothed_predictions[mask],
                                               [dataset.windows[i].group_id for i in result.test_indices[mask]],evaluation_model.classes_,.04)
    (output/'validation_by_condition.json').write_text(json.dumps(by_condition,indent=2)+'\n')
    model=_new_pipeline(7).fit(dataset.data.features[deployment_indices],dataset.data.labels[deployment_indices])
    save_model(model_path,model,window_size=25,hop_size=5,thresholds=0.,sessions=[str(r.path) for r in recordings],
               bandpass_hz=(10.,55.),sample_rate_hz=125.,spatial_reference='common_average',
               baseline_normalize=selected['normalization']=='session_baseline')
    artifact=joblib.load(model_path)
    artifact['compact_profile']={'version':1,'notch_hz':selected['notch_hz'],'notch_q':30.,
                                 'normalization':selected['normalization'],'fixed_reference_scale':scale.tolist(),
                                 'filter_reset_guard_samples':38,'smoothing_windows':7}
    artifact['training_provenance']=config
    joblib.dump(artifact,model_path)
    summary={'model':str(model_path.resolve()),'selected_variant':selected['variant'],'training_trials':len(deployment_ids),
             'training_windows':len(deployment_indices),'internal_validation_smoothed_macro_f1':result.metrics['smoothed']['macro_f1'],
             'internal_validation_balanced_accuracy':result.metrics['smoothed']['balanced_accuracy'],
             'paired_reference_only_macro_f1':reference_result.metrics['smoothed']['macro_f1'],
             'independent_rest_false_active_fraction':result.metrics['independent_rest']['smoothed']['false_active_window_fraction'],
             'unfamiliar_position_accuracy':'not measured; chunk 4 unused'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-output',type=Path,required=True)
    args=parser.parse_args();train(args.collection,args.output,args.model_output)
