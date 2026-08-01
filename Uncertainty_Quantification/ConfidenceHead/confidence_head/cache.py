"""Versioned fail-closed ConfidenceHead feature cache."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator
import torch
from .artifacts import atomic_json_dump, atomic_torch_save, load_torch_artifact
from .features import ContinuousBatch
from .identity import sha256_file
CACHE_SCHEMA_VERSION=1
class CacheCorruptionError(ValueError): pass
class CacheIncompleteError(ValueError): pass
@dataclass(frozen=True)
class CacheShard:
    split:str; shard_index:int; filename:str; sha256:str; num_structures:int; num_atoms:int
@dataclass(frozen=True)
class CacheProgress:
    cache_id:str; shard_max_atoms:int; splits:dict[str,dict[str,Any]]
@dataclass(frozen=True)
class CacheManifest:
    root:Path; cache_id:str; complete:bool; splits:dict[str,tuple[CacheShard,...]]
def _bad(where:str,what:str): return CacheCorruptionError(f"{where}: {what}")
def _tensor(x:object,where:str,dtype=None,ndim=None):
    if not isinstance(x,torch.Tensor) or x.device.type!="cpu" or (dtype is not None and x.dtype!=dtype) or (ndim is not None and x.ndim!=ndim): raise _bad(where,"invalid tensor")
    return x
def _payload(p:object,cache_id:str,split:str,shard:int,start:int,where:str):
    if not isinstance(p,dict): raise _bad(where,"payload is not a mapping")
    keys={"schema_version","cache_id","split","shard_index","num_structures","num_atoms","indices","structure_ids","atom_offsets","features","reference_energy","reference_forces"}
    if set(p)!=keys or p["schema_version"]!=1 or p["cache_id"]!=cache_id or p["split"]!=split or p["shard_index"]!=shard: raise _bad(where,"schema or identity mismatch")
    n=_tensor(p["num_atoms"],where,torch.long,1); ix=_tensor(p["indices"],where,torch.long,1); off=_tensor(p["atom_offsets"],where,torch.long,1); f=_tensor(p["features"],where,ndim=2); e=_tensor(p["reference_energy"],where,ndim=1); force=_tensor(p["reference_forces"],where,ndim=2); ids=tuple(p["structure_ids"])
    if not all(isinstance(x,str) and x for x in ids) or len(ids)!=len(set(ids)): raise _bad(where,"duplicate structure_ids")
    if not isinstance(p["num_structures"],int) or p["num_structures"]<1 or len(ids)!=len(n)!=len(ix)!=p["num_structures"] or not torch.all(n>0): raise _bad(where,"false counts")
    expected=torch.cat((torch.zeros(1,dtype=torch.long),n.cumsum(0)))
    if not torch.equal(off,expected): raise _bad(where,"atom_offsets do not cover structures")
    a=int(off[-1])
    if f.shape!=(a,640) or force.shape!=(a,3) or e.shape!=(len(n),) or not f.is_floating_point() or f.dtype!=e.dtype or f.dtype!=force.dtype: raise _bad(where,"dtype or shape mismatch")
    if not torch.isfinite(f).all() or not torch.isfinite(e).all() or not torch.isfinite(force).all(): raise _bad(where,"tensors must be finite")
    if not torch.equal(ix,torch.arange(start,start+len(n),dtype=torch.long)): raise _bad(where,"indices must be continuous")
    return len(n),a,ids
def _load_progress(root:Path,cache_id:str):
    try: p=load_torch_artifact(root/"progress.pt")
    except Exception as x: raise _bad("progress.pt",f"cannot load safely: {x}") from x
    if not isinstance(p,dict) or p.get("schema_version")!=1 or p.get("cache_id")!=cache_id or not isinstance(p.get("shard_max_atoms"),int) or not isinstance(p.get("splits"),dict): raise _bad("progress.pt","schema or cache_id mismatch")
    return CacheProgress(cache_id,p["shard_max_atoms"],p["splits"])
def _validate(root:Path,progress:CacheProgress):
    for split,state in progress.splits.items():
        if not isinstance(state,dict) or not isinstance(state.get("shards"),list): raise _bad("progress.pt","invalid split state")
        seen=set(); start=atoms=0; names=set()
        for i,record in enumerate(state["shards"]):
            try: sh=CacheShard(**record)
            except Exception as x: raise _bad("progress.pt","invalid shard record") from x
            if sh.split!=split or sh.shard_index!=i: raise _bad(split,"noncontinuous shard indices")
            path=root/split/sh.filename; names.add(sh.filename)
            if not path.is_file(): raise _bad(f"{split}/{sh.filename}","is missing")
            if sha256_file(path)!=sh.sha256: raise _bad(f"{split}/{sh.filename}","hash mismatch")
            try: c,a,ids=_payload(load_torch_artifact(path),progress.cache_id,split,i,start,f"{split}/{sh.filename}")
            except CacheCorruptionError: raise
            except Exception as x: raise _bad(f"{split}/{sh.filename}",f"cannot load safely: {x}") from x
            if (c,a)!=(sh.num_structures,sh.num_atoms) or seen.intersection(ids): raise _bad(f"{split}/{sh.filename}","false counts or duplicate structure_ids")
            seen.update(ids); start+=c; atoms+=a
        if (state.get("next_index"),state.get("num_structures"),state.get("num_atoms"),state.get("next_shard_index"))!=(start,start,atoms,len(names)): raise _bad(split,"false committed counts")
        if (root/split).exists() and {x.name for x in (root/split).glob("shard-*.pt")}!=names: raise _bad(split,"missing or unexpected shard")
class CacheWriter:
 def __init__(self,root:Path,*,cache_id:str,shard_max_atoms:int):
  if not cache_id or shard_max_atoms<1: raise ValueError("invalid cache parameters")
  self.root=Path(root);self.cache_id=cache_id;self.shard_max_atoms=shard_max_atoms;self.root.mkdir(parents=True,exist_ok=True)
  if (self.root/"cache_manifest.json").exists(): raise CacheCorruptionError("complete cache is immutable")
  if (self.root/"progress.pt").exists():
   p=_load_progress(self.root,cache_id);_validate(self.root,p)
   if p.shard_max_atoms!=shard_max_atoms: raise _bad("progress.pt","shard_max_atoms mismatch")
   self.splits=p.splits
  else:self.splits={};self._save()
  self.buf=[];self.active=None;self.seen={}
  for s,st in self.splits.items(): self.seen[s]={i for r in st["shards"] for i in load_torch_artifact(self.root/s/r["filename"])["structure_ids"]}
 @classmethod
 def resume(cls,root:Path,*,expected_cache_id:str):
  p=_load_progress(Path(root),expected_cache_id);_validate(Path(root),p);return cls(root,cache_id=expected_cache_id,shard_max_atoms=p.shard_max_atoms)
 def _save(self): atomic_torch_save(self.root/"progress.pt",{"schema_version":1,"cache_id":self.cache_id,"shard_max_atoms":self.shard_max_atoms,"splits":self.splits})
 def _state(self,s):
  st=self.splits.setdefault(s,{"next_index":0,"num_structures":0,"num_atoms":0,"next_shard_index":0,"shards":[],"complete":False})
  if st["complete"]: raise CacheCorruptionError(f"{s}: split is immutable")
  self.seen.setdefault(s,set());return st
 def append(self,batch:ContinuousBatch,*,split="train"):
  if self.active not in (None,split): raise CacheCorruptionError("cannot interleave splits")
  st=self._state(split);self.active=split
  for j in range(len(batch.structure_ids)):
   a,z=int(batch.atom_offsets[j]),int(batch.atom_offsets[j+1]); one=ContinuousBatch(batch.indices[j:j+1].detach().cpu(),(batch.structure_ids[j],),batch.num_atoms[j:j+1].detach().cpu(),torch.tensor([0,z-a]),batch.features[a:z].detach().cpu(),batch.reference_energy[j:j+1].detach().cpu(),batch.reference_forces[a:z].detach().cpu())
   p={"schema_version":1,"cache_id":self.cache_id,"split":split,"shard_index":st["next_shard_index"],"num_structures":1,"num_atoms":one.num_atoms,"indices":one.indices,"structure_ids":one.structure_ids,"atom_offsets":one.atom_offsets,"features":one.features,"reference_energy":one.reference_energy,"reference_forces":one.reference_forces}
   _payload(p,self.cache_id,split,st["next_shard_index"],st["next_index"]+len(self.buf),split)
   if one.structure_ids[0] in self.seen[split] or any(x.structure_ids[0]==one.structure_ids[0] for x in self.buf): raise _bad(split,"duplicate structure_ids")
   if self.buf and sum(int(x.num_atoms[0]) for x in self.buf)+z-a>self.shard_max_atoms:self._flush(split)
   self.buf.append(one)
 def _flush(self,s):
  if not self.buf:return
  st=self._state(s);n=torch.cat([x.num_atoms for x in self.buf]);p={"schema_version":1,"cache_id":self.cache_id,"split":s,"shard_index":st["next_shard_index"],"num_structures":len(self.buf),"num_atoms":n,"indices":torch.cat([x.indices for x in self.buf]),"structure_ids":tuple(x.structure_ids[0] for x in self.buf),"atom_offsets":torch.cat((torch.zeros(1,dtype=torch.long),n.cumsum(0))),"features":torch.cat([x.features for x in self.buf]),"reference_energy":torch.cat([x.reference_energy for x in self.buf]),"reference_forces":torch.cat([x.reference_forces for x in self.buf])};c,a,ids=_payload(p,self.cache_id,s,st["next_shard_index"],st["next_index"],s);path=self.root/s/f"shard-{st['next_shard_index']:06d}.pt";path.parent.mkdir(parents=True,exist_ok=True);atomic_torch_save(path,p);st["shards"].append(asdict(CacheShard(s,st["next_shard_index"],path.name,sha256_file(path),c,a)));st["next_index"]+=c;st["num_structures"]+=c;st["num_atoms"]+=a;st["next_shard_index"]+=1;self.seen[s].update(ids);self.buf=[];self._save()
 def finalize_split(self,s):
  if self.active not in (None,s): raise CacheCorruptionError("active split mismatch")
  self._state(s);self.active=s;self._flush(s);self.splits[s]["complete"]=True;self._save();self.active=None
 def finalize(self):
  if self.active or not self.splits or not all(x["complete"] for x in self.splits.values()): raise CacheIncompleteError("cache has incomplete splits")
  p=_load_progress(self.root,self.cache_id);_validate(self.root,p);atomic_json_dump(self.root/"cache_manifest.json",{"schema_version":1,"cache_id":self.cache_id,"complete":True,"splits":{s:x["shards"] for s,x in self.splits.items()}});return load_complete_cache(self.root,expected_cache_id=self.cache_id)
def load_complete_cache(root:Path,*,expected_cache_id:str):
 root=Path(root);path=root/"cache_manifest.json"
 if not path.is_file(): raise CacheIncompleteError("cache manifest is missing")
 try:m=json.loads(path.read_text())
 except Exception as x:raise _bad("cache_manifest.json",f"cannot decode: {x}") from x
 if not isinstance(m,dict) or m.get("schema_version")!=1:raise _bad("cache_manifest.json","schema_version is unsupported")
 if m.get("cache_id")!=expected_cache_id:raise _bad("cache_manifest.json","cache_id does not match")
 if m.get("complete") is not True or not isinstance(m.get("splits"),dict):raise CacheIncompleteError("cache manifest is incomplete")
 p=_load_progress(root,expected_cache_id);_validate(root,p)
 if not all(x["complete"] for x in p.splits.values()) or m["splits"]!={s:x["shards"] for s,x in p.splits.items()}:raise _bad("cache_manifest.json","does not match progress")
 return CacheManifest(root,expected_cache_id,True,{s:tuple(CacheShard(**x) for x in r) for s,r in m["splits"].items()})
def iter_cache_batches(manifest:CacheManifest,split:str,batch_size:int)->Iterator[ContinuousBatch]:
 if batch_size<1:raise ValueError("batch_size must be positive")
 pending=[]
 expected=0
 seen=set()
 for sh in manifest.splits.get(split,()):
  path=manifest.root/split/sh.filename
  if not path.is_file() or sha256_file(path)!=sh.sha256:raise _bad(split+"/"+sh.filename,"missing or modified shard")
  try:p=load_torch_artifact(path)
  except Exception as x:raise _bad(split+"/"+sh.filename,f"cannot load safely: {x}") from x
  c,a,ids=_payload(p,manifest.cache_id,split,sh.shard_index,expected,split+"/"+sh.filename)
  if (c,a)!=(sh.num_structures,sh.num_atoms) or seen.intersection(ids):raise _bad(split+"/"+sh.filename,"false counts or duplicate structure_ids")
  expected+=c;seen.update(ids)
  for i,sid in enumerate(p["structure_ids"]):
   a,z=int(p["atom_offsets"][i]),int(p["atom_offsets"][i+1]);pending.append(ContinuousBatch(p["indices"][i:i+1],(sid,),p["num_atoms"][i:i+1],torch.tensor([0,z-a]),p["features"][a:z],p["reference_energy"][i:i+1],p["reference_forces"][a:z]))
   if len(pending)==batch_size:yield _repack(pending);pending=[]
 if pending:yield _repack(pending)
def _repack(x):
 n=torch.cat([i.num_atoms for i in x]);return ContinuousBatch(torch.cat([i.indices for i in x]),tuple(i.structure_ids[0] for i in x),n,torch.cat((torch.zeros(1,dtype=torch.long),n.cumsum(0))),torch.cat([i.features for i in x]),torch.cat([i.reference_energy for i in x]),torch.cat([i.reference_forces for i in x]))
