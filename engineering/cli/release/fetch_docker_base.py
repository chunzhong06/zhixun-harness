"""Windows网络下载官方基础镜像，校验层摘要后交给已有WSL Docker导入。"""
from pathlib import Path
import gzip
import hashlib
import io
import json
import sys
import tarfile
import urllib.request

def main(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    token=json.load(urllib.request.urlopen('https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/python:pull',timeout=30))['token']
    def get(path):
        request=urllib.request.Request('https://registry-1.docker.io/v2/library/python/'+path,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json'})
        with urllib.request.urlopen(request,timeout=120) as r:return r.read()
    index=json.loads(get('manifests/3.13-slim'))
    selected=next(m for m in index['manifests'] if m['platform'].get('os')=='linux' and m['platform'].get('architecture')=='amd64')
    manifest=json.loads(get('manifests/'+selected['digest']))
    def blob(descriptor):
        value=get('blobs/'+descriptor['digest'])
        assert hashlib.sha256(value).hexdigest()==descriptor['digest'].split(':')[1]
        return value
    config=blob(manifest['config']);name=manifest['config']['digest'].split(':')[1]+'.json'
    with tarfile.open(root/'python-base.tar','w') as tar:
        def add(name,value):
            info=tarfile.TarInfo(name);info.size=len(value);tar.addfile(info,io.BytesIO(value))
        add(name,config);layers=[]
        for descriptor in manifest['layers']:
            value=blob(descriptor);value=gzip.decompress(value) if value[:2]==b'\x1f\x8b' else value
            layer=descriptor['digest'].split(':')[1]+'/layer.tar';add(layer,value);layers.append(layer)
            print(json.dumps({'layer':descriptor['digest'],'unpacked_bytes':len(value)}),flush=True)
        add('manifest.json',json.dumps([{'Config':name,'RepoTags':['python:3.13-slim'],'Layers':layers}]).encode())
    (root/'base-source.json').write_text(json.dumps({'registry':'registry-1.docker.io/library/python','platform':'linux/amd64','selected':selected,'manifest':manifest},indent=2))
    print(str(root/'python-base.tar'),flush=True)

if __name__=='__main__':main(sys.argv[1])
