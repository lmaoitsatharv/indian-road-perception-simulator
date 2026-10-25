"""Bounded WSL GPU service for a single authenticated native camera session."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulation_live_protocol import PROTOCOL,send_packet,receive_packet
from setup.prepare_simulation_dataset import save,sha


def run(config_file):
    import numpy as np
    from PIL import Image
    from idd_live_simulation_inference import LiveSimulationRoadSegmenter
    config=json.loads(config_file.read_text())
    output=config_file.parent
    manifest=output / "model_manifest.json"
    initial=sha(ROOT / config["checkpoint"])
    model=LiveSimulationRoadSegmenter(manifest)
    model.warmup()
    weights_before=[hashlib.sha256(w.numpy().tobytes()).hexdigest() for w in model.model.weights]
    (output / "predictions").mkdir(exist_ok=False)
    (output / "probabilities").mkdir(exist_ok=False)
    status={"status":"ready","pid":os.getpid(),"session":config["session"],"runtime":model.runtime,
            "requests":0,"faults":[],"model_control":False,"gpu_memory":model.tf.config.experimental.get_memory_info('GPU:0')}
    save(output / "worker_status.json",status)
    deadline=time.monotonic()+config["service_max_seconds"]
    probes=config.get("fault_probe",False)
    delayed=disconnected=False
    log=(output / "worker_frames.jsonl").open("w",buffering=1)
    try:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(("127.0.0.1",config["port"]))
            server.listen(1)
            server.settimeout(.2)
            save(output / "worker_ready.json",{"session":config["session"],"port":config["port"],"pid":os.getpid()})
            while time.monotonic()<deadline and not (output / "stop_worker.json").exists():
                try:
                    connection,_=server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(3.)
                    connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                    while time.monotonic()<deadline and not (output / "stop_worker.json").exists():
                        try:
                            header,payload=receive_packet(connection)
                        except (ConnectionError,OSError):
                            break
                        if header.get("protocol")!=PROTOCOL or header.get("token")!=config["token"] or header.get("session")!=config["session"]:
                            break
                        if (header.get("source_shape")!=[540,960,3] or len(payload)!=540*960*3
                                or header.get("control_allowed") is not False):
                            break
                        start=time.perf_counter()
                        index=status["requests"]+1
                        if probes and index==45 and not disconnected:
                            disconnected=True
                            status["faults"].append({"kind":"disconnect","frame":header["frame"]})
                            save(output / "worker_status.json",status)
                            break
                        delay=0.
                        if probes and index==20 and not delayed:
                            delayed=True
                            delay=.9
                            status["faults"].append({"kind":"delay","frame":header["frame"],"seconds":delay})
                            time.sleep(delay)
                        raw=np.frombuffer(payload,np.uint8).reshape(540,960,3)
                        if hashlib.sha256(payload).hexdigest()!=header["source_rgb_sha256"]:
                            raise ValueError("Live camera bytes changed in transit")
                        image=np.array(Image.fromarray(raw).resize((512,512),Image.Resampling.BILINEAR))
                        resize_ms=(time.perf_counter()-start-delay)*1000
                        prediction=model.predict_rgb(image)
                        path=output / "predictions" / f"frame_{header['frame']:07d}.png"
                        Image.fromarray(prediction.mask).save(path,compress_level=1)
                        if index%25==0 or delay:
                            np.save(output / "probabilities" / f"frame_{header['frame']:07d}.npy",prediction.probability,allow_pickle=False)
                        response={key:header[key] for key in ("session","frame","simulation_timestamp_s","tick_start_ns",
                            "tick_return_ns","sensor_callback_ns","source_rgb_sha256")}
                        response.update(protocol=PROTOCOL,inference_ms=prediction.inference_ms,
                            resize_ms=resize_ms,injected_delay_ms=delay*1000,request_index=index,
                            input_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                            mask_sha256=hashlib.sha256(prediction.mask.tobytes()).hexdigest(),
                            worker_total_ms=(time.perf_counter()-start)*1000)
                        log.write(json.dumps(response)+"\n")
                        status["requests"]=index
                        try:
                            send_packet(connection,response,image.tobytes()+prediction.mask.tobytes())
                        except OSError:
                            break
                        if index%10==0:
                            save(output / "worker_status.json",status)
    except BaseException as error:
        status.update(status="failed",error=f"{type(error).__name__}: {error}")
        (output / "worker_error.txt").write_text(traceback.format_exc())
        raise
    finally:
        log.close()
        unchanged=initial==sha(ROOT / config["checkpoint"]) and weights_before==[
            hashlib.sha256(w.numpy().tobytes()).hexdigest() for w in model.model.weights]
        status.update(status="completed" if status["status"]!="failed" and unchanged else "failed",model_weights_unchanged=unchanged)
        save(output / "worker_status.json",status)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args()
    run(args.config.resolve())
