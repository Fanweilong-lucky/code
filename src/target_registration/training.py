"""Base Global–Local Model staged training and Proposed Method local-head adaptation, using validation-only selection."""
import csv,json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from .core import global_local_base,proposed
from .data import load_training_protocol

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')

def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def fresh_output(output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Output is not empty: {output}. Use a new directory; existing results are preserved.')
    output.mkdir(parents=True,exist_ok=True)
    return output

def calibrate_base(records):
    conf=np.array([r['confidence_error_px'] for r in records],dtype=np.float64)
    thresholds=np.concatenate([np.linspace(float(conf.min())-1e-6,float(conf.max())+1e-6,121),[float(conf.max())+1e6]])
    best=None
    for t in thresholds:
        selected=[r['fallback_tre'] if r['confidence_error_px']>t else r['full_global_tre'] for r in records]
        score=global_local_base.objective_value(selected,'rmse')
        if best is None or score<best['score']:best=dict(threshold=float(t),score=score)
    gate_best=None
    for gate in np.linspace(.05,.95,91):
        values=[r['fallback_tre'] if r['confidence_error_px']>best['threshold'] else r['target_candidate_tre'] if r['gate']>=gate else r['full_global_tre'] for r in records]
        score=float(np.mean(values))
        if gate_best is None or score<gate_best['score']:gate_best=dict(threshold=float(gate),score=score)
    return dict(source='VALIDATION_ONLY',validation_samples=len(records),confidence_threshold_error_px=best['threshold'],gate_threshold=gate_best['threshold'])

def train_base(config,output,repository_root,device):
    pairs,counts=load_training_protocol(config,repository_root)
    out=fresh_output(output);device=torch.device(device);global_local_base.set_seed(20260812)
    prior=global_local_base.train_prior_statistics(pairs['train']);pos=global_local_base.compute_gate_pos_weight(pairs['train'])
    model=global_local_base.BaseGlobalLocalRegistrationNet(prior['base_angle_deg'],prior['base_scale_x'],prior['base_scale_y']).to(device)
    train=global_local_base.make_loader(pairs['train'],augment=True,shuffle=True,batch_size=2)
    val=global_local_base.make_loader(pairs['val'],augment=False,shuffle=False,batch_size=2)
    amp=device.type=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp)
    history=[];epoch=0
    for stage,epochs,lr in [('A',45,2e-4),('B',25,2e-4),('C',15,4e-5)]:
        global_local_base.configure_stage(model,stage)
        optimizer=torch.optim.AdamW(global_local_base.trainable_parameters(model),lr=lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
        best=float('inf')
        for stage_epoch in range(1,epochs+1):
            epoch+=1;global_local_base.set_stage_train_mode(model,stage);running={};seen=0
            for batch in train:
                batch=global_local_base.move_batch_to_device(batch,device);optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',enabled=amp):
                    pred=model(batch['visible'],batch['infrared'],batch['bbox'],batch['has_bbox'],enable_local=stage!='A')
                    losses=global_local_base.registration_losses(batch['visible'],pred,batch,stage,pos);loss=losses['total']
                assert torch.isfinite(loss),'Non-finite Base Global–Local Model loss'
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(global_local_base.trainable_parameters(model),5.)
                scaler.step(optimizer);scaler.update()
                bs=batch['visible'].shape[0];seen+=bs
                for k,v in losses.items():running[k]=running.get(k,0.)+float(v.detach())*bs
            scheduler.step();metrics=global_local_base.validate(model,val,device,stage,pos)
            row=dict(global_epoch=epoch,stage=stage,stage_epoch=stage_epoch,lr=optimizer.param_groups[0]['lr'],
                **{'train_'+k:v/seen for k,v in running.items()},**{'val_'+k:v for k,v in metrics.items()})
            history.append(row)
            # Loss dictionaries may gain local terms between stages.
            fields=list(dict.fromkeys(k for r in history for k in r))
            with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(history)
            payload=global_local_base.checkpoint_payload(model,prior,counts,epoch,stage,metrics)
            torch.save(payload,out/'last.pt')
            if metrics['total']<best:
                best=metrics['total'];torch.save(payload,out/f'best_stage_{stage}.pt')
                if stage=='C':torch.save(payload,out/'best_joint.pt')
            print(json.dumps(dict(epoch=epoch,stage=stage,val_loss=metrics['total'])),flush=True)
        model.load_state_dict(torch.load(out/f'best_stage_{stage}.pt',map_location='cpu',weights_only=True)['model'])
    model.load_state_dict(torch.load(out/'best_joint.pt',map_location='cpu',weights_only=True)['model']);model.eval()
    with torch.inference_mode():records=global_local_base.collect_eval_records(model,val,device,prior)
    write_json(out/'calibration_base.json',calibrate_base(records))
    write_json(out/'training_protocol.json',dict(stage_epochs=[45,25,15],seed=20260812,external_pretrained=False,test_images_read=False,counts=counts))

def train_local(config,output,repository_root,device,initialization,calibration):
    pairs,counts=load_training_protocol(config,repository_root)
    out=fresh_output(output);device=torch.device(device)
    ckpt=torch.load(initialization,map_location='cpu',weights_only=True)
    cal=json.loads(Path(calibration).read_text(encoding='utf-8'))
    assert cal['source']=='VALIDATION_ONLY'
    proposed.TRAINING_OUTPUT_ROOT=out;proposed.BASE_CHECKPOINT=Path(initialization)
    model,prior,best,_=proposed.train_seed(20260812,pairs['train'],pairs['val'],ckpt,device,cal['confidence_threshold_error_px'])
    torch.save(dict(model=model.state_dict(),prior_stats=prior,epoch=best['epoch'],seed=20260812),out/'proposed_final.pt')
    write_json(out/'calibration_proposed.json',best['calibration'])
    write_json(out/'training_protocol.json',dict(epochs=15,batch_size=2,seed=20260812,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,
        initialization='trained Base Global–Local Model checkpoint; only target_local_corr is trainable',validation_only_selection=True,test_images_read=False,counts=counts))
