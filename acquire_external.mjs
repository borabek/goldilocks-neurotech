import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
const root=path.dirname(new URL(import.meta.url).pathname),dir=path.join(root,'data/external_sleepedf');
const selection=JSON.parse(await fs.readFile(path.join(dir,'selection.json'),'utf8'));
const sums=Object.fromEntries((await fs.readFile(path.join(dir,'SHA256SUMS.txt'),'utf8')).trim().split('\n').map(line=>{const m=line.trim().split(/\s+/);return[m[1],m[0]]}));
const labels=process.argv.includes('--labels');
if(labels&&!await fs.stat(path.join(root,'results/v2/freeze_record.json')).catch(()=>null))throw Error('External reference labels remain sealed until model freeze.');
const manifest=[];let cursor=0;
async function worker(){while(cursor<selection.records.length){let item=selection.records[cursor++];let name=labels?item.hypnogram:item.psg;let expected=sums['sleep-cassette/'+name];if(!expected)throw Error('Missing published hash '+name);let target=path.join(dir,name);let bytes=await fs.readFile(target).catch(()=>null);let url='https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'+name;if(!bytes||crypto.createHash('sha256').update(bytes).digest('hex')!==expected){let last;for(let attempt=0;attempt<4;attempt++){try{let r=await fetch(url,{signal:AbortSignal.timeout(180000)});if(!r.ok)throw Error(String(r.status));bytes=Buffer.from(await r.arrayBuffer());if(crypto.createHash('sha256').update(bytes).digest('hex')!==expected)throw Error('Checksum mismatch');await fs.writeFile(target,bytes);last=null;break;}catch(e){last=e;}}if(last)throw last;}manifest.push({subject:item.subject,file:name,bytes:bytes.length,sha256:expected,url,reference:labels,acquired_unix:Date.now()/1000});console.log(JSON.stringify({done:manifest.length,total:selection.records.length,file:name}));}}
await Promise.all(Array.from({length:4},worker));
await fs.writeFile(path.join(dir,labels?'label_acquisition.json':'signal_acquisition.json'),JSON.stringify(manifest,null,2));
