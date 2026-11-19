import sys, time, subprocess, numpy as np, torch
sys.path.insert(0, '.')
from tnet import TransNetV2, predict
m = TransNetV2()
sd = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
print(m.load_state_dict(sd, strict=False))
m.eval()
torch.set_num_threads(4)
for src in sys.argv[2:]:
    raw = subprocess.run(['ffmpeg','-v','error','-i',src,'-vf','scale=48:27','-pix_fmt','rgb24','-f','rawvideo','-'],capture_output=True).stdout
    fr = torch.from_numpy(np.frombuffer(raw,np.uint8).reshape(-1,27,48,3).copy())
    t=time.time(); p = predict(m, fr).numpy(); dt=time.time()-t
    starts=[]; prev=False
    for i,v in enumerate(p):
        on=v>0.5
        if prev and not on: starts.append(i)
        prev=on
    np.save(src.split('/')[-1]+'.npy', p)
    print(src.split('/')[-1], len(fr), f"{len(fr)/dt:.0f} im/s", ' '.join(map(str,starts)))
