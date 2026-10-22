"""Score a frozen compact model on a separately recorded evaluation chunk."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))

import numpy as np
from emg.compact_model import read_recording,make_dataset
from emg.evaluation.benchmark import DEFAULT_CONFIG,write_report
from emg.evaluation.dataset_v2 import score_v2
from emg.model import load_model


def evaluate(model_path,session_path,output):
    output=Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory.')
    artifact=load_model(model_path);profile=artifact['compact_profile']
    record=read_recording(session_path,purpose='held_out_evaluation')
    provenance=artifact['training_provenance']
    if record.metadata['session_id'] in {r['session_id'] for r in provenance['sources']}:
        raise ValueError('Evaluation session appears in model training provenance.')
    for key in ('electrode_placement_id','subject_id','source','value_unit','recording_date'):
        original=json.loads((Path(provenance['sources'][0]['source_path'])/'metadata.json').read_text())
        if record.metadata[key]!=original[key]:
            raise ValueError(f'This same-application test has a different {key}.')
    data,conditions,_=make_dataset([record],notch_hz=profile['notch_hz'],normalization=profile['normalization'],
                                  fixed_scale=profile['fixed_reference_scale'])
    classification=np.array([i for i,w in enumerate(data.windows) if w.phase=='gesture'])
    rest=np.array([i for i,w in enumerate(data.windows) if w.phase in ('rest_pre','rest_post')])
    result=score_v2(artifact['model'],data,classification,rest,DEFAULT_CONFIG)
    trial_scores=[]
    for trial_id in sorted({data.windows[i].trial_id for i in classification}):
        mask=np.array([data.windows[i].trial_id==trial_id for i in result.test_indices])
        true=str(data.data.labels[result.test_indices[mask]][0])
        predictions=result.smoothed_predictions[mask]
        labels,counts=np.unique(predictions,return_counts=True)
        majority=str(labels[np.argmax(counts)])
        trial_scores.append({'trial_id':trial_id,'true_label':true,'majority_prediction':majority,
                             'majority_correct':majority==true,'correct_window_fraction':float(np.mean(predictions==true))})
    config={**DEFAULT_CONFIG,'experiment':'frozen_compact_unfamiliar_position',
            'model_sha256':hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
            'model_path':str(Path(model_path).resolve()),'processing_profile':profile,'source':record.audit,
            'training_trial_ids':provenance['deployment_training_trial_ids'],
            'test_trial_ids':[r['trial_id'] for r in trial_scores],
            'model_refitted':False,'hyperparameters_retuned':False,'trial_scores':trial_scores,
            'limitation':'Small same-application static-position test; does not validate arbitrary motion, new electrode placement, or another day.'}
    write_report(output,result,data,config,artifact['model'])
    summary={'smoothed_macro_f1':result.metrics['smoothed']['macro_f1'],
             'balanced_accuracy':result.metrics['smoothed']['balanced_accuracy'],
             'test_trials':len(trial_scores),'majority_correct_trials':sum(t['majority_correct'] for t in trial_scores),
             'independent_rest_false_active_fraction':result.metrics['independent_rest']['smoothed']['false_active_window_fraction'],
             'model_changed':False}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();evaluate(args.model,args.session,args.output)
