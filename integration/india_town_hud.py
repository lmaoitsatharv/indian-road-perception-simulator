"""Presentation HUD for the connected Indian town viewer (display only, never recorded as model input)."""
from __future__ import annotations

import collections
import math

import pygame

COLORS = {"market_street":(236,170,76),"divided_main_road":(92,170,224),"rural_road":(118,190,108),
    "residential_colony":(196,150,214),"irregular_junction":(236,112,98)}
NAMES = {"market_street":"Bazaar market","divided_main_road":"Divided main road","rural_road":"Village roads",
    "residential_colony":"Residential colony","irregular_junction":"Irregular junction"}
STATUS = {"district_speed":("Cruising",(120,214,150)),"vehicle_headway":("Following the vehicle ahead",(240,200,96)),
    "emergency_gap":("Braking for a close vehicle",(240,112,96)),"junction_wait":("Waiting for its turn at the junction",(240,200,96)),
    "urban_bus_stop":("Stopping at the bus stop",(150,190,240)),
    "pedestrian_crossing":("Slowing for a pedestrian crossing",(240,200,96)),
    "emergency_pedestrian":("Braking hard for a pedestrian",(240,112,96)),
    "oncoming_vehicle":("Holding back for oncoming traffic",(240,200,96)),
    "merging":("Merging back into the lane",(150,190,240)),
    "filtering":("Filtering past slower traffic",(120,214,150)),
    "passing":("Squeezing past a two-wheeler",(120,214,150)),
    "overtaking":("Changing lanes to pass a stopped vehicle",(120,214,150)),
    "yield_to_camera_car":("Giving way to your car",(240,200,96)),
    # Camera-only driver (india_camera_driver.REASONS)
    "cruise":("Cruising on the road it sees",(120,214,150)),
    "starting":("Waiting for the first camera result",(150,190,240)),
    "wide_road_area":("Wide road area: slowing down",(240,200,96)),
    "uncertain_road":("Road edge unclear: slowing down",(240,200,96)),
    "lost_road":("Left kerb lost: creeping on",(240,200,96)),
    "lost_road_stop":("Lost the road: stopped",(240,112,96)),
    "curve":("Slowing for the curve",(150,190,240)),
    "vehicle_ahead":("Vehicle ahead on its path",(240,200,96)),
    "person_ahead":("Person ahead on its path",(240,200,96)),
    "person_near_path":("Person close to its path",(240,200,96)),
    "road_ends":("Road ahead not visible",(240,200,96)),
    "emergency_vehicle":("Braking hard for a vehicle",(240,112,96)),
    "emergency_person":("Braking hard for a person",(240,112,96)),
    "perception_unavailable":("Perception unavailable: stopping",(240,112,96)),
    "backing_out":("Backing out of a blocked pass",(150,190,240)),
    "backing_wait":("Waiting to back up",(240,200,96)),
    "shifting_gear":("Stopping to change direction",(240,200,96)),
    "junction_turn":("Turning at the junction (planned path)",(150,190,240)),
    "junction_fallback":("Turning at the junction (route line)",(240,200,96)),
    "waiting_turn":("Waiting for its turn at the junction",(240,200,96)),
    "arriving":("Arriving at the destination",(150,190,240)),
    "arrived":("Arrived",(120,214,150)),
    "uturn_ahead":("Slowing to turn round",(150,190,240)),
    "uturn_wait":("Waiting for a gap to turn round",(240,200,96)),
    "uturn_forward":("Turning round: forward",(150,190,240)),
    "uturn_reverse":("Turning round: backing up",(150,190,240)),
    "no_route":("No route to the destination",(240,112,96))}
DECISION_COLORS = {"PROCEED":(120,214,150),"SLOW":(240,200,96),"STOP":(240,112,96)}
PLACES = tuple(COLORS)
HELP = (("TAB","Chase / driver camera (back to your car)"),
    ("1  2  3  4  5","Street camera: bazaar, main road, village, colony, junction"),
    ("L","Next location"),("V   SHIFT+V","Watch the next / previous vehicle"),("C","Back to your car"),
    ("+   -","Faster / slower: ½×, 1×, 2×, max"),("0","Back to real-time speed"),
    ("SPACE","Pause / resume"),(".","Step one frame while paused"),("R","Reset your car onto its lane"),("W","Change weather"),
    ("A","Camera-driving runs: full-screen AI view"),
    ("M","Camera-driving runs: pick a destination on the town map"),("X","Camera-driving runs: clear the destination"),
    ("H","Hide / show the HUD"),("F","Fullscreen"),("S","Save a snapshot"),("ESC","Quit"))
TEXT,DIM,FAINT = (236,240,236),(176,190,186),(124,140,138)
PANEL,EDGE = (12,18,22,200),(255,255,255,30)
SIZE = (1280,720)


def pretty(name):
    return "your car" if name == "ego" else name.replace("_"," ").capitalize()


ROLE_COLORS = {"in_path":(255,72,60),"near_path":(255,196,64),"ignored":(170,176,180),"passing":(110,170,255),
    "behind":(255,196,64)}
_MASK_INDEX = {}


def ai_view_surface(rgb,mask,plan,geometry,scale=.5,rear=None,insets=None):
    """Annotated camera picture for the HUD: road mask (green), kerb points (white),
    planned path (cyan) and detector boxes (red = braking for it, amber = near its
    path, blue = the stopped vehicle being passed, grey = ignored). While the rear and
    side cameras run, their pictures and boxes sit along the top, as mirrors would:
    left, rear, right. insets: {view: (picture, detections)}; rear=picture is the
    rear one with plan.rear. Display only; drawn at half size by default (the panel
    is 384x216), so the overlay costs little per frame."""
    import numpy as np
    step = int(round(1/scale))
    rgb = rgb[::step,::step]
    height,width = rgb.shape[:2]
    if (height,width) not in _MASK_INDEX:
        _MASK_INDEX[(height,width)] = ((np.arange(height)*512//height)[:,None],(np.arange(width)*512//width)[None,:])
    rows,cols = _MASK_INDEX[(height,width)]
    picture = rgb.copy()
    if mask is not None:
        road = mask[rows,cols] == 1
        picture[road] = (picture[road]*.6+np.array([40,220,120])*.4).astype(np.uint8)
    surface = pygame.surfarray.make_surface(picture.swapaxes(0,1))
    if plan is None:
        return surface
    k = 1/step   # camera pixels -> this picture's pixels
    if plan.path is not None:
        xs = np.linspace(max(2.6,geometry.nearest_visible_ground()),16.,40)
        ys = np.interp(xs,plan.path[:,0],plan.path[:,1])
        u,v,valid = geometry.project(xs,ys)
        points = [(float(a)*k,float(b)*k) for a,b,ok in zip(u,v,valid) if ok]
        if len(points) > 1:
            pygame.draw.lines(surface,(20,20,24),False,points,5)
            pygame.draw.lines(surface,(80,230,255),False,points,3)
    edges = [(x,y) for x,y,_,_ in plan.rows if y is not None]
    asphalt = set(getattr(plan,"asphalt_rows",[]) or [])
    if edges:
        u,v,valid = geometry.project(np.array([e[0] for e in edges]),np.array([e[1] for e in edges]))
        for (x,_),a,b,ok in zip(edges,u,v,valid):
            if ok:
                # amber: the edge was moved in from a shoulder to where the asphalt starts
                pygame.draw.circle(surface,(255,196,64) if x in asphalt else (255,255,255),(int(a*k),int(b*k)),3)
                pygame.draw.circle(surface,(20,20,24),(int(a*k),int(b*k)),3,1)
    font = _font(12,True)
    for box in getattr(plan,"rejected",[]):
        x1,y1,x2,y2 = (value*k for value in box["box"])
        pygame.draw.rect(surface,(150,150,160),pygame.Rect(int(x1),int(y1),max(1,int(x2-x1)),max(1,int(y2-y1))),1)
        surface.blit(font.render(f"ignored {box['class']} (road visible under it)",True,(170,170,180)),(int(x1)+2,int(y1)+2))
    for obstacle in plan.obstacles:
        x1,y1,x2,y2 = (value*k for value in obstacle["box"])
        color = ROLE_COLORS.get(obstacle["role"],(170,176,180))
        pygame.draw.rect(surface,color,pygame.Rect(int(x1),int(y1),max(1,int(x2-x1)),max(1,int(y2-y1))),2 if obstacle["role"] != "ignored" else 1)
        distance = obstacle["distance"] if obstacle["distance"] is not None else max(0.,obstacle["x"])
        name = obstacle['class'].replace('_',' ')
        label = font.render(f"passing {name}" if obstacle["role"] == "passing" else f"{name} {distance:.0f} m",True,color)
        back = pygame.Surface((label.get_width()+4,label.get_height()),pygame.SRCALPHA)
        back.fill((0,0,0,150))
        surface.blit(back,(int(x1),max(0,int(y1)-label.get_height())))
        surface.blit(label,(int(x1)+2,max(0,int(y1)-label.get_height())))
    junction_path = getattr(plan,"junction_path",None)
    if junction_path is not None and len(junction_path) > 1:
        ahead = junction_path[junction_path[:,0] > max(2.6,geometry.nearest_visible_ground())]
        if len(ahead) > 1:
            u,v,valid = geometry.project(ahead[:,0],ahead[:,1])
            points = [(float(a)*k,float(b)*k) for a,b,ok in zip(u,v,valid) if ok]
            if len(points) > 1:
                pygame.draw.lines(surface,(20,20,24),False,points,6)
                pygame.draw.lines(surface,(236,72,220),False,points,4)   # magenta: the planned junction path
        goal = getattr(plan,"junction_goal",None)
        if goal is not None:
            u,v,valid = geometry.project(np.array([goal[0]]),np.array([goal[1]]))
            if valid[0]:
                pygame.draw.circle(surface,(236,72,220),(int(u[0]*k),int(v[0]*k)),6,2)
    if plan.road_end_x is not None:
        x = plan.road_end_x
        y = float(np.interp(x,plan.path[:,0],plan.path[:,1])) if plan.path is not None else 0.
        u,v,valid = geometry.project(np.array([x,x]),np.array([y-1.2,y+1.2]))
        if valid.all():
            pygame.draw.line(surface,(255,72,60),(u[0]*k,v[0]*k),(u[1]*k,v[1]*k),3)
    insets = dict(insets or {})
    if rear is not None:
        insets["rear"] = (rear,getattr(plan,"rear",None) or [])
    # Half the picture's width alone; with two or three running, a third of the width each (they sit in fixed
    # places, left, rear in the middle, right, so they do not move as the cameras take turns).
    across = width//2 if len(insets) == 1 else (width-24)//3
    for view,(shot,found) in insets.items():
        # The picture, with what the detector found there (rear: metres behind; side: metres away).
        inset = pygame.transform.smoothscale(pygame.surfarray.make_surface(shot[::step,::step].swapaxes(0,1)),
            (across,round(across*shot.shape[0]/shot.shape[1])))
        q = across/shot.shape[1]
        for obstacle in found:
            x1,y1,x2,y2 = (value*q for value in obstacle["box"])
            pygame.draw.rect(inset,ROLE_COLORS["behind"],pygame.Rect(int(x1),int(y1),max(1,int(x2-x1)),max(1,int(y2-y1))),2)
            metres = obstacle["behind_m"] if view == "rear" else obstacle["distance"]
            inset.blit(font.render(f"{obstacle['class'].replace('_',' ')} {metres:.0f} m",True,ROLE_COLORS["behind"]),
                (int(x1)+2,max(0,int(y1)-14)))
        x = {"left":6,"rear":(width-inset.get_width())//2,"right":width-inset.get_width()-6}[view]
        place = (x,6)
        pygame.draw.rect(surface,(20,20,24),pygame.Rect(place[0]-2,place[1]-2,inset.get_width()+4,inset.get_height()+4))
        surface.blit(inset,place)
        label = font.render(view.upper(),True,(255,255,255))   # at the bottom: box labels sit at the top
        surface.blit(label,(place[0]+4,place[1]+inset.get_height()-label.get_height()-2))
    note = getattr(plan,"pass_note","")
    if note:
        text = font.render(note[:70],True,(255,255,255))
        back = pygame.Surface((text.get_width()+8,text.get_height()+4),pygame.SRCALPHA)
        back.fill((0,0,0,170))
        surface.blit(back,(4,height-text.get_height()-8))
        surface.blit(text,(8,height-text.get_height()-6))
    return surface


def _font(size,bold=False):
    for name in ("segoeui","arial"):
        if name in pygame.font.get_fonts():
            return pygame.font.SysFont(name,size,bold=bold)
    return pygame.font.Font(None,int(size*1.3))


class TownHud:
    def __init__(self,map_base,map_xy,map_offset):
        self.fonts = {"tiny":_font(13),"small":_font(15),"body":_font(17),"bodyb":_font(17,True),
            "title":_font(25,True),"speed":_font(58,True),"big":_font(40,True),"label":_font(12,True)}
        self.cache = {}
        self.map_base,self.map_xy,self.map_offset = map_base,map_xy,map_offset
        self.trail = collections.deque(maxlen=90)
        self.trail_t = -1e9
        self.toast = None
        self.flash_until = 0.
        self.district = None
        self.rate = collections.deque(maxlen=60)
        self.panels = {}

    def text(self,value,font,color=TEXT):
        key = (value,font,color)
        if key not in self.cache:
            if len(self.cache) > 600:
                self.cache.clear()
            self.cache[key] = self.fonts[font].render(value,True,color)
        return self.cache[key]

    def panel(self,surface,rect):
        key = (rect.w,rect.h)
        if key not in self.panels:
            p = pygame.Surface(key,pygame.SRCALPHA)
            pygame.draw.rect(p,PANEL,p.get_rect(),border_radius=14)
            pygame.draw.rect(p,EDGE,p.get_rect(),1,border_radius=14)
            self.panels[key] = p
        surface.blit(self.panels[key],rect)

    def notify(self,message,color,now,seconds=3.2):
        self.toast = (message,color,now,seconds)

    def snapshot(self,now):
        self.flash_until = now+.25
        self.notify("Snapshot saved",(230,230,230),now,1.6)

    def draw(self,surface,view,*,trace,sim_time,wall_now,vehicles,people,visited,travelled,paused,camera,
            focus="ego",place=None,place_xy=None,speed=1.,weather="clear",show_hud=True,show_help=False,ai=None,nav=None):
        surface.fill((10,15,18))
        if ai is not None and ai.get("big") and ai.get("view") is not None:
            view = pygame.transform.smoothscale(ai["view"],SIZE)
        if view is not None:
            surface.blit(view,(0,0))
        rows = {row["name"]:row for row in trace}
        ego = rows["ego"]
        watched = rows.get(focus,ego)
        district = place or watched["district"]
        if district != self.district:
            if self.district is not None and place is None:
                self.notify("Entering  "+NAMES[district],COLORS[district],wall_now)
            self.district = district
        self.rate.append((wall_now,sim_time))
        if sim_time-self.trail_t > .5:
            self.trail.append((ego["x"],ego["y"]))
            self.trail_t = sim_time
        if place is not None:
            subtitle = "Live street camera  ·  press C for your car"
        elif focus != "ego":
            subtitle = "Watching "+pretty(focus)+"  ·  press C for your car"
        elif ai is not None:
            subtitle = "Camera-only AI driver  ·  road model + detector in control"
        else:
            subtitle = "Rule-based driver  ·  no trained model in control"
        if show_hud:
            self._header(surface,district,subtitle)
            self._clock(surface,sim_time,camera,speed,weather)
            self._minimap(surface,trace,watched["name"],place,place_xy,nav)
            self._speed(surface,watched,watched.get("distance_m",travelled),focus != "ego" and place is None)
            self._areas(surface,visited,vehicles,people)
            self._keys(surface)
            if ai is not None:
                self._ai_panel(surface,ai,ego,nav)
        else:
            hint = self.text("H  show HUD","tiny",FAINT)
            surface.blit(hint,hint.get_rect(bottomright=(SIZE[0]-14,SIZE[1]-10)))
        self._toast(surface,wall_now)
        if wall_now < self.flash_until:
            veil = pygame.Surface(SIZE,pygame.SRCALPHA)
            veil.fill((255,255,255,int(150*(self.flash_until-wall_now)/.25)))
            surface.blit(veil,(0,0))
        if paused and not show_help:
            veil = pygame.Surface(SIZE,pygame.SRCALPHA)
            veil.fill((6,10,12,140))
            surface.blit(veil,(0,0))
            label = self.text("PAUSED","big")
            surface.blit(label,label.get_rect(center=(SIZE[0]//2,SIZE[1]//2-14)))
            hint = self.text("SPACE  resume     .  step one frame","body",DIM)
            surface.blit(hint,hint.get_rect(center=(SIZE[0]//2,SIZE[1]//2+28)))
        if show_help:
            self._help(surface)

    def _header(self,surface,district,subtitle):
        rect = pygame.Rect(18,16,372,92)
        self.panel(surface,rect)
        surface.blit(self.text("ONE INDIAN TOWN","label",FAINT),(rect.x+18,rect.y+12))
        pygame.draw.circle(surface,COLORS[district],(rect.x+26,rect.y+46),7)
        surface.blit(self.text(NAMES[district],"title"),(rect.x+42,rect.y+28))
        surface.blit(self.text(subtitle,"tiny",DIM),(rect.x+18,rect.y+66))

    def _clock(self,surface,sim_time,camera,speed,weather):
        rate = None
        if len(self.rate) > 10 and self.rate[-1][0]-self.rate[0][0] > .5:
            rate = (self.rate[-1][1]-self.rate[0][1])/(self.rate[-1][0]-self.rate[0][0])
        minutes,seconds = divmod(int(sim_time),60)
        parts = [f"{minutes:02d}:{seconds:02d}"]
        if rate is not None:
            parts.append(f"{rate:.2f}× real time")
        if speed != 1.:
            parts.append("max speed" if speed is None else f"speed {speed:g}×")
        parts.append(weather)
        parts.append(camera)
        label = self.text("   ·   ".join(parts),"small",DIM)
        rect = label.get_rect(midtop=(SIZE[0]//2-70,22)).inflate(30,14)
        self.panel(surface,rect)
        surface.blit(label,label.get_rect(center=rect.center))

    def _minimap(self,surface,trace,focus,place,place_xy,nav=None):
        ox,oy = self.map_offset
        w,h = self.map_base.get_size()
        self.panel(surface,pygame.Rect(ox-10,oy-10,w+20,h+20))
        surface.blit(self.map_base,(ox,oy))
        if nav is not None and nav.get("route_xy") is not None and len(nav["route_xy"]) > 1:
            route = [(ox+p[0],oy+p[1]) for p in (self.map_xy(x,y) for x,y in nav["route_xy"][::6])]
            if len(route) > 1:
                pygame.draw.lines(surface,(20,20,24),False,route,5)
                pygame.draw.lines(surface,(80,230,255),False,route,3)
        if nav is not None and nav.get("pin") is not None:
            x,y = self.map_xy(*nav["pin"])
            pygame.draw.circle(surface,(18,22,24),(ox+x,oy+y),7)
            pygame.draw.circle(surface,(236,72,220),(ox+x,oy+y),5)
        points = [(ox+p[0],oy+p[1]) for p in (self.map_xy(x,y) for x,y in self.trail)]
        if len(points) > 1:
            pygame.draw.lines(surface,(255,244,210),False,points,2)
        for row in trace:
            if row["name"] == "ego":
                continue
            x,y = self.map_xy(row["x"],row["y"])
            pygame.draw.circle(surface,(18,22,24),(ox+x,oy+y),4)
            pygame.draw.circle(surface,(250,150,112),(ox+x,oy+y),3)
            if row["name"] == focus:
                pygame.draw.circle(surface,(255,255,255),(ox+x,oy+y),8,2)
        if place is not None and place_xy is not None:
            x,y = self.map_xy(*place_xy)
            pygame.draw.circle(surface,(18,22,24),(ox+x,oy+y),9)
            pygame.draw.circle(surface,COLORS[place],(ox+x,oy+y),7)
            pygame.draw.circle(surface,(255,255,255),(ox+x,oy+y),7,2)
        ego = next(row for row in trace if row["name"] == "ego")
        x,y = self.map_xy(ego["x"],ego["y"])
        c,s = math.cos(ego["yaw"]),math.sin(ego["yaw"])
        glow = pygame.Surface((44,44),pygame.SRCALPHA)
        pygame.draw.circle(glow,(255,255,255,60),(22,22),20)
        pygame.draw.circle(glow,(255,255,255,110),(22,22),12)
        surface.blit(glow,(ox+x-22,oy+y-22))
        tip = (ox+x+c*13,oy+y+s*13)
        left = (ox+x-c*7-s*8,oy+y-s*7+c*8)
        right = (ox+x-c*7+s*8,oy+y-s*7-c*8)
        pygame.draw.polygon(surface,(255,255,255),[tip,left,right],0)
        pygame.draw.polygon(surface,(18,22,24),[tip,left,right],2)

    def _speed(self,surface,ego,travelled,other):
        rect = pygame.Rect(18,SIZE[1]-150,330,132)
        self.panel(surface,rect)
        if other:
            tag = self.text(pretty(ego["name"]).upper(),"label",FAINT)
            surface.blit(tag,(rect.right-18-tag.get_width(),rect.y+4))
        kmh = ego["speed"]*3.6
        number = self.text(f"{kmh:.0f}","speed")
        surface.blit(number,(rect.x+18,rect.y+2))
        surface.blit(self.text("km/h","bodyb",DIM),(rect.x+26+number.get_width(),rect.y+44))
        target = ego.get("target",0.)*3.6
        surface.blit(self.text(f"target {target:.0f}","small",FAINT),(rect.right-96,rect.y+20))
        surface.blit(self.text(f"{travelled/1000:.2f} km driven","small",FAINT),(rect.right-120,rect.y+44))
        bar = pygame.Rect(rect.x+18,rect.y+84,rect.w-36,6)
        top = 50.
        pygame.draw.rect(surface,(48,58,60),bar,border_radius=3)
        fill = bar.copy()
        fill.w = int(bar.w*min(1.,kmh/top))
        if fill.w > 0:
            pygame.draw.rect(surface,(120,214,150),fill,border_radius=3)
        mark = bar.x+int(bar.w*min(1.,target/top))
        pygame.draw.line(surface,(255,255,255),(mark,bar.y-4),(mark,bar.bottom+3),2)
        if ego.get("holds_junction") is not None:
            label,color = "Crossing the junction (has its turn)",(120,214,150)
        elif kmh < .5 and ego.get("reason") == "district_speed":
            label,color = "Pulling away",(150,190,240)
        else:
            label,color = STATUS.get(ego.get("reason"),(str(ego.get("reason","")).replace("_"," ").capitalize(),DIM))
        pygame.draw.circle(surface,color,(rect.x+24,rect.y+109),5)
        surface.blit(self.text(label,"body"),(rect.x+36,rect.y+98))

    def _ai_panel(self,surface,ai,ego,nav=None):
        """What the camera driver sees and decides (its own camera, road mask, boxes, path), and
        where the sat-nav is taking it."""
        big = ai.get("big")
        rect = pygame.Rect(18,118,404,320) if not big else pygame.Rect(18,118,404,104)
        self.panel(surface,rect)
        surface.blit(self.text("WHAT THE AI SEES" if not big else "AI VIEW  ·  A to return","label",FAINT),(rect.x+14,rect.y+10))
        y = rect.y+30
        if not big:
            if ai.get("view") is not None:
                surface.blit(pygame.transform.smoothscale(ai["view"],(384,216)),(rect.x+10,y))
            else:
                pygame.draw.rect(surface,(29,37,49),(rect.x+10,y,384,216))
                label = self.text("Waiting for the first camera result","small",DIM)
                surface.blit(label,label.get_rect(center=(rect.x+202,y+108)))
            y += 222
        decision = ego.get("decision") or "STOP"
        color = DECISION_COLORS.get(decision,DIM)
        chip = pygame.Rect(rect.x+12,y+2,74,20)
        pygame.draw.rect(surface,color,chip,border_radius=5)
        label = self.text(decision,"label",(20,26,28))
        surface.blit(label,label.get_rect(center=chip.center))
        detail = ego.get("detail") or ""
        surface.blit(self.text(detail[:44],"small"),(chip.right+8,y+3))
        timing = ai.get("timing") or ""
        surface.blit(self.text(timing,"tiny",DIM),(rect.x+14,y+26))
        if nav is not None and nav.get("destination"):
            parts = [f"To {nav['destination']}"]
            if nav.get("remaining_m") is not None and nav.get("next") != "Arrived":
                parts.append(f"{nav['remaining_m']:.0f} m")
            if nav.get("next"):
                parts.append(nav["next"] if nav["next"] == "Arrived" else "next: "+nav["next"])
            surface.blit(self.text("  ·  ".join(parts)[:62],"small",(236,180,230)),(rect.x+14,y+44))
        else:
            surface.blit(self.text("No destination  ·  M to pick one on the map","small",FAINT),(rect.x+14,y+44))

    def _areas(self,surface,visited,vehicles,people):
        rect = pygame.Rect(SIZE[0]-268,SIZE[1]-196,250,178)
        self.panel(surface,rect)
        surface.blit(self.text(f"{vehicles} vehicles  ·  {people} people","bodyb"),(rect.x+16,rect.y+12))
        surface.blit(self.text(f"AREAS VISITED  {len(visited)}/5","label",FAINT),(rect.x+16,rect.y+40))
        for i,(key,color) in enumerate(COLORS.items()):
            y = rect.y+62+i*22
            done = key in visited
            if done:
                pygame.draw.circle(surface,color,(rect.x+22,y+9),6)
            else:
                pygame.draw.circle(surface,color,(rect.x+22,y+9),6,2)
            surface.blit(self.text(NAMES[key],"small",TEXT if done else FAINT),(rect.x+36,y))

    def _keys(self,surface):
        hints = [("TAB","camera"),("1-5","places"),("V","cars"),("+ -","speed"),("SPACE","pause"),("R","reset car"),("F1","all keys")]
        pieces = []
        for key,meaning in hints:
            pieces.append((self.text(key,"label",(20,26,28)),self.text(meaning,"small",DIM)))
        width = sum(k.get_width()+m.get_width()+34 for k,m in pieces)+12
        rect = pygame.Rect(0,0,width,34)
        rect.midbottom = (SIZE[0]//2+40,SIZE[1]-18)
        self.panel(surface,rect)
        x = rect.x+14
        for k,m in pieces:
            chip = pygame.Rect(x,rect.y+8,k.get_width()+12,18)
            pygame.draw.rect(surface,(222,228,224),chip,border_radius=5)
            surface.blit(k,k.get_rect(center=chip.center))
            surface.blit(m,(chip.right+7,rect.y+7))
            x = chip.right+7+m.get_width()+16

    def _help(self,surface):
        veil = pygame.Surface(SIZE,pygame.SRCALPHA)
        veil.fill((6,10,12,170))
        surface.blit(veil,(0,0))
        rect = pygame.Rect(0,0,660,30+len(HELP)*27+56)
        rect.center = (SIZE[0]//2,SIZE[1]//2)
        self.panel(surface,rect)
        surface.blit(self.text("KEYS","label",FAINT),(rect.x+28,rect.y+20))
        for i,(key,meaning) in enumerate(HELP):
            y = rect.y+46+i*27
            label = self.text(key,"label",(20,26,28))
            chip = pygame.Rect(rect.x+28,y,max(34,label.get_width()+14),20)
            pygame.draw.rect(surface,(222,228,224),chip,border_radius=5)
            surface.blit(label,label.get_rect(center=chip.center))
            surface.blit(self.text(meaning,"body"),(rect.x+180,y-1))
        hint = self.text("F1 or ESC to close","small",FAINT)
        surface.blit(hint,hint.get_rect(midbottom=(rect.centerx,rect.bottom-14)))

    def _toast(self,surface,now):
        if self.toast is None:
            return
        message,color,start,seconds = self.toast
        age = now-start
        if age > seconds:
            self.toast = None
            return
        alpha = min(1.,age/.25,(seconds-age)/.6)
        label = self.fonts["title"].render(message,True,TEXT)
        box = pygame.Surface((label.get_width()+56,58),pygame.SRCALPHA)
        pygame.draw.rect(box,PANEL,box.get_rect(),border_radius=16)
        pygame.draw.rect(box,color+(255,),pygame.Rect(0,0,8,58),border_top_left_radius=16,border_bottom_left_radius=16)
        box.blit(label,(30,(58-label.get_height())//2))
        box.set_alpha(int(255*alpha))
        surface.blit(box,box.get_rect(midtop=(SIZE[0]//2-70,128-int(10*(1-alpha)))))


class MapPicker:
    """Full-screen town map for picking a destination: click a street point or a place name.
    World coordinates are CARLA's (metres); the picture keeps north-up as the minimap does."""

    def __init__(self,lines,places,size=SIZE):
        self.lines,self.places,self.size = lines,dict(places),size
        xs = [x for _,points in lines for x,_ in points]+[x for x,_ in self.places.values()]
        ys = [y for _,points in lines for _,y in points]+[y for _,y in self.places.values()]
        self.low,self.high = (min(xs),min(ys)),(max(xs),max(ys))
        w,h = size
        self.top,self.bottom = 70,50
        self.scale = min((w-80)/max(1.,self.high[0]-self.low[0]),(h-self.top-self.bottom)/max(1.,self.high[1]-self.low[1]))
        self.centre = ((self.low[0]+self.high[0])/2,(self.low[1]+self.high[1])/2)

    def to_pixel(self,x,y):
        w,h = self.size
        mid = self.top+(h-self.top-self.bottom)/2
        return int(round(w/2+(x-self.centre[0])*self.scale)),int(round(mid+(y-self.centre[1])*self.scale))

    def to_world(self,px,py):
        w,h = self.size
        mid = self.top+(h-self.top-self.bottom)/2
        return self.centre[0]+(px-w/2)/self.scale,self.centre[1]+(py-mid)/self.scale

    def _label_rect(self,name):
        px,py = self.to_pixel(*self.places[name])
        return pygame.Rect(px+9,py-10,7*len(name)+10,20)

    def label_pixel(self,name):
        return self._label_rect(name).center

    def hit_place(self,px,py):
        for name in self.places:
            x,y = self.to_pixel(*self.places[name])
            if self._label_rect(name).collidepoint(px,py) or math.hypot(px-x,py-y) <= 10:
                return name
        return None

    def draw(self,surface,car_xy,route_xy,pin,status,car_yaw=0.):
        veil = pygame.Surface(self.size,pygame.SRCALPHA)
        veil.fill((8,12,16,236))
        surface.blit(veil,(0,0))
        title = _font(25,True).render("Pick a destination",True,TEXT)
        surface.blit(title,(30,18))
        hint = _font(15).render("Click a street or a place name   ·   M or Esc to close   ·   X clears the destination",True,DIM)
        surface.blit(hint,(32+title.get_width()+18,26))
        for _,points in self.lines:
            pixels = [self.to_pixel(x,y) for x,y in points]
            if len(pixels) > 1:
                pygame.draw.lines(surface,(70,84,90),False,pixels,7)
                pygame.draw.lines(surface,(150,160,160),False,pixels,3)
        if route_xy is not None and len(route_xy) > 1:
            pixels = [self.to_pixel(x,y) for x,y in route_xy[::4]]
            if len(pixels) > 1:
                pygame.draw.lines(surface,(80,230,255),False,pixels,5)
        label = _font(15,True)
        for name,(x,y) in self.places.items():
            px,py = self.to_pixel(x,y)
            pygame.draw.circle(surface,(18,22,24),(px,py),7)
            pygame.draw.circle(surface,(240,200,96),(px,py),5)
            rect = self._label_rect(name)
            back = pygame.Surface(rect.size,pygame.SRCALPHA)
            back.fill((0,0,0,150))
            surface.blit(back,rect)
            surface.blit(label.render(name,True,(240,226,190)),(rect.x+5,rect.y+2))
        if pin is not None:
            px,py = self.to_pixel(*pin)
            pygame.draw.circle(surface,(18,22,24),(px,py),11)
            pygame.draw.circle(surface,(236,72,220),(px,py),8)
        if car_xy is not None:
            px,py = self.to_pixel(*car_xy)
            c,s = math.cos(car_yaw),math.sin(car_yaw)
            tip,left,right = (px+c*14,py+s*14),(px-c*8-s*9,py-s*8+c*9),(px-c*8+s*9,py-s*8-c*9)
            pygame.draw.polygon(surface,(255,255,255),[tip,left,right],0)
            pygame.draw.polygon(surface,(18,22,24),[tip,left,right],2)
        if status:
            text = _font(17).render(status,True,TEXT)
            surface.blit(text,(32,self.size[1]-38))
