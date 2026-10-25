"""Native camera transport and frame-aligned UI. Never reads/writes a driver."""
import hashlib
import json
from pathlib import Path
import socket
import sys
import threading
import time

import numpy as np
import pygame

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulation_live_protocol import PROTOCOL,LatestSlot,freshness,receive_packet,send_packet


class LiveRoadPreview:
    def __init__(self,config_file,episode):
        self.config=json.loads(Path(config_file).read_text())
        self.session=self.config["session"]
        owner=json.loads((ROOT / "setup/carla/server_owner.json").read_text())
        if (Path(owner["output"])!=episode.parent or owner["status"]!="running"
                or self.config["model_control"] is not False):
            raise ValueError("Live preview requires this owned native simulator session")
        self.folder=episode / "live_road"
        self.folder.mkdir(exist_ok=False)
        self.frames_log=(self.folder / "receipts.jsonl").open("w",buffering=1)
        self.display_log=(self.folder / "display.jsonl").open("w",buffering=1)
        self.tick_log=(self.folder / "source_ticks.jsonl").open("w",buffering=1)
        self.error_log=(self.folder / "transport_errors.jsonl").open("w",buffering=1)
        self.slots=LatestSlot()
        self.lock=threading.Lock()
        self.ticks={}
        self.pending=self.latest_rgb=self.result=None
        self.connected=False
        self.closed=threading.Event()
        self.connection=None
        self.current_frame,self.current_sim=0,0.
        self.counters={"camera_frames":0,"unregistered_frames":0,"responses":0,"connection_errors":0}
        self.saved_states=set()
        self.pause_probe_start=None
        self.pause_probe_done=False
        self.thread=threading.Thread(target=self.transport,name="live-road-loopback",daemon=True)
        self.thread.start()

    def receive_rgb(self,image):
        if self.closed.is_set():
            return
        callback_ns=time.perf_counter_ns()
        rgb=np.frombuffer(image.raw_data,np.uint8).reshape(image.height,image.width,4)[...,2::-1].copy()
        frame={"frame":image.frame,"simulation_timestamp_s":image.timestamp,
               "sensor_callback_ns":callback_ns,"rgb":rgb}
        with self.lock:
            self.counters["camera_frames"]+=1
            self.latest_rgb=frame
            self.pending=frame
            self.publish_pending()

    def publish_pending(self):
        # Called only while holding the small metadata lock.
        if self.pending is None:
            return
        clock=self.ticks.get(self.pending["frame"])
        if clock:
            self.slots.put({**self.pending,**clock})
            self.pending=None
        elif self.pending["frame"]<self.current_frame-3:
            self.counters["unregistered_frames"]+=1
            self.pending=None

    def register_tick(self,frame,simulation_timestamp_s,start_ns,return_ns):
        clock={"tick_start_ns":start_ns,"tick_return_ns":return_ns}
        with self.lock:
            self.current_frame,self.current_sim=frame,simulation_timestamp_s
            self.ticks[frame]=clock
            for old in list(self.ticks):
                if old<frame-100:
                    del self.ticks[old]
            self.publish_pending()
        self.tick_log.write(json.dumps({"frame":frame,"simulation_timestamp_s":simulation_timestamp_s,**clock})+"\n")

    def transport(self):
        while not self.closed.is_set():
            first_frame=self.slots.take()
            if first_frame is None:
                continue
            operation="connect"
            try:
                connection=socket.create_connection(("127.0.0.1",self.config["port"]),timeout=3.)
                connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                self.connection=connection
                with connection:
                    with self.lock:
                        self.connected=True
                    while not self.closed.is_set():
                        frame=first_frame if first_frame is not None else self.slots.take()
                        first_frame=None
                        if frame is None:
                            continue
                        payload=frame["rgb"].tobytes()
                        header={k:v for k,v in frame.items() if k!="rgb"}
                        header.update(protocol=PROTOCOL,session=self.session,token=self.config["token"],
                            source_shape=[540,960,3],source_rgb_sha256=hashlib.sha256(payload).hexdigest(),control_allowed=False)
                        sent_ns=time.perf_counter_ns()
                        operation="send_request"
                        send_packet(connection,header,payload)
                        operation="receive_response"
                        response,data=receive_packet(connection)
                        operation="validate_response"
                        received_ns=time.perf_counter_ns()
                        for key in ("session","frame","tick_start_ns","tick_return_ns","sensor_callback_ns",
                                    "source_rgb_sha256","simulation_timestamp_s"):
                            if response.get(key)!=header[key]:
                                raise ValueError("Live response identity mismatch: "+key)
                        if response.get("protocol")!=PROTOCOL or len(data)!=512*512*4:
                            raise ValueError("Unexpected live response shape")
                        rgb=np.frombuffer(data[:512*512*3],np.uint8).reshape(512,512,3).copy()
                        mask=np.frombuffer(data[512*512*3:],np.uint8).reshape(512,512).copy()
                        if (np.any(mask>1) or hashlib.sha256(mask.tobytes()).hexdigest()!=response["mask_sha256"]
                                or hashlib.sha256(rgb.tobytes()).hexdigest()!=response["input_rgb_sha256"]):
                            raise ValueError("Live response pixels/hash differ")
                        row={**response,"sent_ns":sent_ns,"received_ns":received_ns,
                             "callback_to_result_ms":(received_ns-header["sensor_callback_ns"])/1e6,
                             "source_step_to_result_upper_ms":(received_ns-header["tick_start_ns"])/1e6,
                             "roundtrip_ms":(received_ns-sent_ns)/1e6}
                        with self.lock:
                            self.result={"header":row,"rgb":rgb,"mask":mask}
                            self.counters["responses"]+=1
                        self.frames_log.write(json.dumps(row,allow_nan=False)+"\n")
            except (OSError,ConnectionError,ValueError) as error:
                if not self.closed.is_set():
                    self.error_log.write(json.dumps({"operation":operation,"error":str(error),
                        "type":type(error).__name__,"at_ns":time.perf_counter_ns(),"current_frame":self.current_frame})+"\n")
                with self.lock:
                    self.connected=False
                    self.result=None
                    self.counters["connection_errors"]+=1
                self.closed.wait(.2)
            finally:
                self.connection=None

    def probe_events(self,ticks,paused):
        if not self.config.get("fault_probe") or self.pause_probe_done:
            return
        if ticks>=240 and self.pause_probe_start is None and not paused:
            self.pause_probe_start=time.perf_counter()
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_SPACE))
        elif self.pause_probe_start is not None and paused and time.perf_counter()-self.pause_probe_start>=1.2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_SPACE))
            self.pause_probe_done=True

    def draw(self,display,font,viewer_surface):
        with self.lock:
            latest,result,connected=self.latest_rgb,self.result,self.connected
            current_sim,current_frame=self.current_sim,self.current_frame
        header=result["header"] if result else None
        state=freshness(header,now_ns=time.perf_counter_ns(),current_sim_s=current_sim,
            connected=connected,session=self.session,
            wall_limit_ms=self.config["max_age_ms"]-self.config.get("presentation_reserve_ms",100),
            sim_limit_ms=self.config["max_sim_age_ms"])
        if not connected and header is None and self.counters["responses"]==0:
            state["state"]="waiting"
        shown=state["state"]=="fresh"
        display.blit(font.render("Live front camera",True,(235,235,235)),(10,5))
        display.blit(font.render("Matching-frame road preview",True,(235,235,235)),(490,5))
        if latest:
            display.blit(pygame.transform.smoothscale(pygame.surfarray.make_surface(latest["rgb"].swapaxes(0,1)),(480,270)),(0,32))
            display.blit(font.render(f"Camera frame {latest['frame']}",True,(195,210,230)),(10,310))
        if shown:
            overlay=result["rgb"].copy()
            road=result["mask"]==1
            overlay[road]=(overlay[road]*.55+np.array([30,230,130])*.45).astype(np.uint8)
            display.blit(pygame.transform.smoothscale(pygame.surfarray.make_surface(overlay.swapaxes(0,1)),(480,270)),(480,32))
        else:
            pygame.draw.rect(display,(29,37,49),(480,32,480,270))
            label={"stale":"Result too old - mask hidden","offline":"Service offline - mask hidden"}.get(state["state"],"Waiting for a fresh road mask")
            display.blit(font.render(label,True,(255,205,130)),(500,150))
        age="--" if state["age_upper_ms"] is None else f"{state['age_upper_ms']:.0f} ms"
        frame="--" if not header else str(header["frame"])
        display.blit(font.render(f"{state['state'].upper()} | source {frame} | age <= {age}",True,(185,235,185)),(490,310))
        if viewer_surface:
            display.blit(pygame.transform.smoothscale(viewer_surface,(360,202)),(0,338))
        lines=["Road preview only; no model vehicle control",
               "Age bound includes rendering, transfer and model",
               "Mask is drawn on its own source image",
               f"Freshness limit: {self.config['max_age_ms']} ms wall / {self.config['max_sim_age_ms']} ms simulation",
               "Green: predicted road; no object detection yet"]
        if header:
            lines.append(f"GPU inference {header['inference_ms']:.1f} ms")
        if self.config.get("fault_probe"):
            lines.append("Controlled delay/disconnect/pause tests active")
        for index,line in enumerate(lines):
            display.blit(font.render(line,True,(190,210,230)),(375,347+25*index))
        return {"header":header,"shown":shown,"state_at_draw":state["state"],
                "current_sim_s":current_sim,"current_frame":current_frame,"connected":connected}

    def presented(self,display,draw_state):
        now=time.perf_counter_ns()
        state=freshness(draw_state["header"],now_ns=now,current_sim_s=draw_state["current_sim_s"],
            connected=draw_state["connected"],session=self.session,wall_limit_ms=self.config["max_age_ms"],
            sim_limit_ms=self.config["max_sim_age_ms"])
        header=draw_state["header"]
        row={**state,"state_at_draw":draw_state["state_at_draw"],"overlay_shown":draw_state["shown"],
             "presented_ns":now,"source_frame":header["frame"] if header else None,
             "source_tick_start_ns":header["tick_start_ns"] if header else None,
             "simulation_timestamp_s":header["simulation_timestamp_s"] if header else None,
             "current_sim_s":draw_state["current_sim_s"],"current_frame":draw_state["current_frame"],
             "session":self.session,"model_control":False}
        self.display_log.write(json.dumps(row,allow_nan=False)+"\n")
        label=draw_state["state_at_draw"]
        if label not in self.saved_states:
            self.saved_states.add(label)
            pygame.image.save(display,str(self.folder / (label+".png")))
        if draw_state["shown"] and state["state"]!="fresh":
            self.counters["presentation_deadline_violations"]=self.counters.get("presentation_deadline_violations",0)+1

    def finish(self):
        self.closed.set()
        self.slots.close()
        connection=self.connection
        if connection:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(timeout=2.)
        for stream in (self.frames_log,self.display_log,self.tick_log,self.error_log):
            stream.close()
        (self.folder / "summary.json").write_text(json.dumps({**self.counters,
            "queue_replacements":self.slots.replaced,"pending_capacity":1,"inflight_capacity":1,
            "thread_stopped":not self.thread.is_alive(),"states_shown":sorted(self.saved_states),
            "model_control":False,"wall_limit_ms":self.config["max_age_ms"],
            "sim_limit_ms":self.config["max_sim_age_ms"],"pause_probe_completed":self.pause_probe_done},indent=2)+"\n")
        if self.thread.is_alive():
            raise RuntimeError("Live observer transport thread did not close")
