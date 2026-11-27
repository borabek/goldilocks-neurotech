import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import path from 'node:path';
const root=path.dirname(path.dirname(new URL(import.meta.url).pathname));
const pin='face09565c4337ddbce7a048c9b02060ad20df45';
const data=path.join(root,'data','boas');
await fs.mkdir(data,{recursive:true});
const tree=JSON.parse(await fs.readFile(path.join(root,'data/source_tree.json'),'utf8'));
const manifest=JSON.parse(await fs.readFile(path.join(root,'data','boas_acquisition.json'),'utf8').catch(()=> '[]'));
async function fetchBytes(url){let last;for(let attempt=0;attempt<4;attempt++){try{const r=await fetch(url,{signal:AbortSignal.timeout(180000)});if(!r.ok)throw Error(`${r.status} ${url}`);return Buffer.from(await r.arrayBuffer());}catch(e){last=e;}}throw last;}
const meta=tree.tree.filter(x=>['README','dataset_description.json','participants.tsv'].includes(x.path)||(/headband/.test(x.path)&&/\.(json|tsv)$/.test(x.path))||(/acq-psg_events.tsv$/.test(x.path)));
let cursor=0;
async function metaWorker(){while(cursor<meta.length){const x=meta[cursor++];const dest=path.join(data,x.path);await fs.mkdir(path.dirname(dest),{recursive:true});if(!await fs.stat(dest).catch(()=>null))await fs.writeFile(dest,await fetchBytes(`https://raw.githubusercontent.com/OpenNeuroDatasets/ds005555/${pin}/${x.path}`));}}
await Promise.all(Array.from({length:6},metaWorker));
console.log(JSON.stringify({metadataFiles:meta.length}));
const edfs=tree.tree.filter(x=>/acq-headband_eeg.edf$/.test(x.path)).sort((a,b)=>Number(a.path.match(/sub-(\d+)/)[1])-Number(b.path.match(/sub-(\d+)/)[1]));
cursor=0;
async function worker(){while(cursor<edfs.length){const x=edfs[cursor++];const pointer=(await fetchBytes(`https://raw.githubusercontent.com/OpenNeuroDatasets/ds005555/${pin}/${x.path}`)).toString();const m=pointer.match(/SHA256E-s(\d+)--([a-f0-9]{64})/);if(!m)throw Error('Invalid annex pointer '+x.path);const expectedSize=Number(m[1]),expectedHash=m[2];const dest=path.join(data,x.path);const compact=path.join(root,'data/eeg_digital',x.path.split('/')[0]+'.npz');if(await fs.stat(compact).catch(()=>null)){continue;}while((await fs.statfs(root)).bavail*(await fs.statfs(root)).bsize<2.5e9){await new Promise(resolve=>setTimeout(resolve,1000));}const url=`https://s3.amazonaws.com/openneuro.org/ds005555/${x.path}`;let bytes=await fs.readFile(dest).catch(()=>null);if(!bytes||bytes.length!==expectedSize||crypto.createHash('sha256').update(bytes).digest('hex')!==expectedHash){bytes=await fetchBytes(url);if(bytes.length!==expectedSize||crypto.createHash('sha256').update(bytes).digest('hex')!==expectedHash)throw Error('Checksum mismatch '+x.path);await fs.writeFile(dest+'.part',bytes);await fs.rename(dest+'.part',dest);}const previous=manifest.findIndex(v=>v.path===x.path);if(previous>=0)manifest.splice(previous,1);manifest.push({path:x.path,url,bytes:bytes.length,sha256:expectedHash,pin});await fs.writeFile(path.join(root,'data','boas_acquisition.json'),JSON.stringify(manifest,null,2));console.log(JSON.stringify({downloaded:manifest.length,total:edfs.length,file:x.path,bytes:bytes.length}));}}
await Promise.all(Array.from({length:4},worker));
await fs.writeFile(path.join(root,'data','boas_acquisition.json'),JSON.stringify(manifest.sort((a,b)=>a.path.localeCompare(b.path)),null,2));
console.log('BOAS_COMPLETE');
