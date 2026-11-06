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
    "overtaking":("Changing lanes to pass a stopped vehicle",(120,214,150))}
PLACES = tuple(COLORS)
HELP = (("TAB","Chase / driver camera (back to your car)"),
    ("1  2  3  4  5","Street camera: bazaar, main road, village, colony, junction"),
    ("L","Next location"),("V   SHIFT+V","Watch the next / previous vehicle"),("C","Back to your car"),
    ("+   -","Faster / slower: ½×, 1×, 2×, max"),("0","Back to real-time speed"),
    ("SPACE","Pause / resume"),(".","Step one frame while paused"),("R","Reset your car onto its lane"),("W","Change weather"),
    ("H","Hide / show the HUD"),("F","Fullscreen"),("S","Save a snapshot"),("ESC","Quit"))
TEXT,DIM,FAINT = (236,240,236),(176,190,186),(124,140,138)
PANEL,EDGE = (12,18,22,200),(255,255,255,30)
SIZE = (1280,720)


def pretty(name):
    return "your car" if name == "ego" else name.replace("_"," ").capitalize()


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
            focus="ego",place=None,place_xy=None,speed=1.,weather="clear",show_hud=True,show_help=False):
        surface.fill((10,15,18))
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
        else:
            subtitle = "Rule-based driver  ·  no trained model in control"
        if show_hud:
            self._header(surface,district,subtitle)
            self._clock(surface,sim_time,camera,speed,weather)
            self._minimap(surface,trace,watched["name"],place,place_xy)
            self._speed(surface,watched,watched.get("distance_m",travelled),focus != "ego" and place is None)
            self._areas(surface,visited,vehicles,people)
            self._keys(surface)
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

    def _minimap(self,surface,trace,focus,place,place_xy):
        ox,oy = self.map_offset
        w,h = self.map_base.get_size()
        self.panel(surface,pygame.Rect(ox-10,oy-10,w+20,h+20))
        surface.blit(self.map_base,(ox,oy))
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
