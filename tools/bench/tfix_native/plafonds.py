# Plafonds théoriques de TemporalFix (docs/reports/tfix-natif/plafonds.md) : max(FLOP / débit, octets / bande) couche par couche,
# RTX 4070 Ti, formes de s2. Usage : python3 tools/bench/tfix_native/plafonds.py
import math
cd=lambda a,b:-(-a//b)
F16,F32,FP8,BW=160e12,80e12,321e12,504e9
def conv(name,h,w,cin,cout,stride,batch,sep=False,fold=False):
    oh,ow=cd(h,stride),cd(w,stride)
    if sep: mac=oh*ow*(9*cin+cin*cout)*batch  # vrai réseau : par canal puis 1x1
    else: mac=oh*ow*9*cin*cout*batch
    folded=oh*ow*9*cin*cout*batch
    byts=2*(h*w*cin+oh*ow*cout)*batch+2*9*cin*cout
    return dict(name=name,oh=oh,ow=ow,flop=2*mac,ffold=2*folded,bytes=byts)
def ceil(l,rate):
    return max(l['flop']/rate,l['bytes']/BW)
def net(H,W):
    enc=[]
    h,w=H,W
    enc.append(conv('enc 3→16 p2',h,w,3,16,2,1)); h,w=cd(h,2),cd(w,2)
    enc+= [conv('enc 16→16',h,w,16,16,1,1),conv('enc 16→16',h,w,16,16,1,1),conv('enc 16→4',h,w,16,4,1,1)]
    win=[]
    blocks=[(0,15,48,96,False),(1,26,32,64,True),(2,26,24,48,True)]
    for b,ci,c1,c2,sep in blocks:
        s=8>>b; h,w=cd(H,s),cd(W,s)
        win.append(conv(f'bloc {b} {ci}→{c1} p2',h,w,ci,c1,2,6)); h,w=cd(h,2),cd(w,2)
        win.append(conv(f'bloc {b} {c1}→{c2} p2',h,w,c1,c2,2,6)); h,w=cd(h,2),cd(w,2)
        for i in range(4): win.append(conv(f'bloc {b} ResConv {c2}'+(' sép.' if sep else ''),h,w,c2,c2,1,6,sep=sep))
        win.append(conv(f'bloc {b} {c2}→11',h,w,c2,11,1,6))
    h,w=cd(H,2),cd(W,2)
    win+=[conv('tête 25→32',h,w,25,32,1,6),conv('tête 32→32',h,w,32,32,1,6),conv('tête 32→1',h,w,32,1,1,6)]
    return enc,win
def glue_bytes(H,W):
    # colle minimale : X1, X2 et entrée de la tête écrites (26, 26, 25 canaux x6), pyramides lues (7 canaux,
    # 7 images), images du réseau lues pour la tête et le mélange (3 canaux x 7, deux fois), sortie écrite.
    b=0
    for l in (1,2):
        s=8>>l; p=cd(H,s)*cd(W,s); b+=2*6*26*p+2*7*7*p
    p2=cd(H,2)*cd(W,2); b+=2*6*25*p2+2*7*3*H*W*2+2*3*H*W
    return b
for label,(H,W) in (('demi-taille (réseau 960×540)',(540,960)),('pleine taille (réseau 1920×1080)',(1080,1920))):
    enc,win=net(H,W)
    print('##',label)
    for part,L in (('encodeur',enc),('fenêtre',win)):
        tf=sum(l['flop'] for l in L); tff=sum(l['ffold'] for l in L); tb=sum(l['bytes'] for l in L)
        c16=sum(ceil(l,F16) for l in L); c32=sum(ceil(l,F32) for l in L); c8=sum(ceil(l,FP8) for l in L)
        print(part, 'GFLOP %.1f (replié %.1f)  Mo %.0f  plafond f16 %.3f f32 %.3f fp8 %.3f ms'%(tf/1e9,tff/1e9,tb/1e6,c16*1e3,c32*1e3,c8*1e3))
        for l in L: print('   %-24s %4dx%-4d GFLOP %6.2f (replié %6.2f) Mo %6.1f  f16 %.3f ms  f32 %.3f ms  %s'%(l['name'],l['ow'],l['oh'],l['flop']/1e9,l['ffold']/1e9,l['bytes']/1e6,ceil(l,F16)*1e3,ceil(l,F32)*1e3,'octets' if l['bytes']/BW>l['flop']/F16 else 'calcul'))
    gb=glue_bytes(H,W); print('colle (glue, conf, mélange) Mo %.0f -> %.3f ms'%(gb/1e6,gb/BW*1e3))
    # tout fondu : entrées (pyramides 7 images niveaux 0-2, images réseau 7) lues une fois, sortie écrite, calcul total
    p=sum(cd(H,8>>l)*cd(W,8>>l) for l in range(3))
    fb=2*7*7*p+2*7*3*H*W+2*3*H*W
    tf=sum(l['flop'] for l in win)
    print('fenêtre tout fondue: Mo %.0f, %.3f ms octets, %.3f ms calcul f16, %.3f f32'%(fb/1e6,fb/BW*1e3,tf/F16*1e3,tf/F32*1e3))
