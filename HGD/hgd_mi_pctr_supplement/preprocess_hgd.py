"""Fixed 2a-like preprocessing, 44 HGD channels; no outcome-driven choices."""
import argparse,json,time
from pathlib import Path
import mne,numpy as np,scipy.io as sio
from scipy.signal import butter,filtfilt,iirnotch
from DataLoader.LoadData import euclidean_alignment
ROOT=Path('/home/u24600002150531/8410/data/BCI/HGD')
CHANNELS=['FC5','FC1','FC2','FC6','C3','C4','CP5','CP1','CP2','CP6','FC3','FCz','FC4','C5','C1','C2','C6','CP3','CPz','CP4','FFC5h','FFC3h','FFC4h','FFC6h','FCC5h','FCC3h','FCC4h','FCC6h','CCP5h','CCP3h','CCP4h','CCP6h','CPP5h','CPP3h','CPP4h','CPP6h','FFC1h','FFC2h','FCC1h','FCC2h','CCP1h','CCP2h','CPP1h','CPP2h']
EVENTS={'left_hand':1,'right_hand':2,'feet':3,'rest':4}
def process(s,split):
    dest=ROOT/'processed';record=dest/f'S{s}{split}_audit.json'
    if record.exists():
        prior=json.loads(record.read_text())
        for output in prior['outputs']:
            p=Path(output['path']);assert p.is_file()
        return prior
    rawfile=ROOT/'raw'/split/f'{s}.edf'
    raw=mne.io.read_raw_edf(rawfile,infer_types=True,preload=False,verbose=False)
    missing=set(CHANNELS)-set(raw.ch_names);assert not missing,missing
    raw.pick(CHANNELS);raw.load_data();raw.reorder_channels(CHANNELS)
    desc=set(raw.annotations.description);assert desc==set(EVENTS),desc
    # Same physical passband/order and notch as original 2a loader; no HGD-specific gamma extension.
    sf=float(raw.info['sfreq']);b,a=butter(4,[0.1,40.],btype='bandpass',fs=sf)
    raw._data=filtfilt(b,a,raw._data,axis=-1)
    b,a=iirnotch(50.,30.,sf);raw._data=filtfilt(b,a,raw._data,axis=-1)
    raw.resample(250,verbose=False)
    ev,_=mne.events_from_annotations(raw,event_id=EVENTS,verbose=False)
    ep=mne.Epochs(raw,ev,event_id=EVENTS,tmin=-.5,tmax=4.-1/250,baseline=None,preload=True,proj=False,reject_by_annotation=True,event_repeated='error',verbose=False)
    x=ep.get_data()*1e6;y=ep.events[:,-1]-1
    assert x.shape[1:]==(44,1125),x.shape
    assert np.isfinite(x).all() and set(y)=={0,1,2,3}
    outputs=[]
    for name,arr in [('HGD_noea',x),('HGD_ea',euclidean_alignment(x))]:
        p=dest/name/f'S{s}{split}'/'data.mat';p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():raise RuntimeError(f'Unverified existing cache: {p}; refusing overwrite')
        tmp=p.with_name('data.pending.mat');sio.savemat(tmp,{'x_data':arr.astype(np.float32),'y_data':y.astype(np.int64)},do_compression=False);tmp.rename(p)
        outputs.append({'path':str(p)})
    row={'subject':s,'split':split,'shape':list(x.shape),'class_counts':np.bincount(y,minlength=4).tolist(),'class_order':list(EVENTS),'annotation_count':len(ev),'epoch_drop_log':[list(z) for z in ep.drop_log],'channels':CHANNELS,'filter':'Butterworth4 .1-40Hz zero-phase + 50Hz Q30 notch, resample250','window':[-.5,3.996],'EA':'per subject and split; test EEG statistics accessible, not labels; same original preprocessing permission','outputs':outputs}
    record.write_text(json.dumps(row,indent=2));print('PREPROCESSED',s,split,x.shape,np.bincount(y),flush=True);return row
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--subjects',nargs='+',type=int,default=list(range(1,15)));ap.add_argument('--splits',nargs='+',default=['train','test']);args=ap.parse_args()
    (ROOT/'processed').mkdir(parents=True,exist_ok=True)
    rows=[process(s,split) for s in args.subjects for split in args.splits]
    if args.subjects==list(range(1,15)) and args.splits==['train','test']:
        assert len(rows)==28
        (ROOT/'processed'/'preprocessing_complete.json').write_text(json.dumps({'files':rows,'completed_at':time.strftime('%Y-%m-%d %H:%M:%S')},indent=2))
if __name__=='__main__':main()
