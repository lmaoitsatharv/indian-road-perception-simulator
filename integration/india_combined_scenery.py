"""All five scenery districts alive together in the connected native town."""
import math
import random
from types import SimpleNamespace

import carla
import numpy as np

from india_surface import MESH,asphalt_texture
from india_world_scenery import WorldScenery,HOUSE,APARTMENT,ASSETS,image_texture
from india_combined_map import BY_ID
from india_combined_traffic import jaywalk_spots


class CombinedScenery(WorldScenery):
    def __init__(self,world,owned,xml,network):
        super().__init__(world,owned,"market_street",xml)
        self.network = network
        self.current_street = None

    def _record(self,actor,kind,project_class,collision,**detail):
        row = super()._record(actor,kind,project_class,collision,**detail)
        if self.current_street is not None:
            row.update(street_id=self.current_street.id,district=self.current_street.district)
        return row

    # Background tree rows behind the frontage, so the town is not an empty
    # plain. Presentation only: optional props, never on or beside a driving lane.
    TREE_SPACING = {"market_street":36.,"residential_colony":24.,"divided_main_road":28.,"rural_road":18.,"irregular_junction":26.}

    def greenery(self,street,spec,lane):
        spacing = self.TREE_SPACING[street.district]
        for row,(depth,step) in enumerate(((21.,spacing),(33.,spacing*1.7))):
            for k,s in enumerate(np.arange(26.+row*7.,spec["length"]-26.,step)):
                wp = self.map.get_waypoint_xodr(street.id,lane,float(s))
                outer = wp.lane_width/2+street.sidewalk+street.shoulder
                jitter = ((k*37+row*11+street.id*7)%9)-4.
                location = self.outside(wp,outer+depth+jitter)
                if any(self._on_drive(location.x+dx,location.y+dy) for dx in (-6.,0.,6.) for dy in (-6.,0.,6.)):
                    continue
                palm = street.district in ("rural_road","residential_colony","irregular_junction") and (k+row)%3 != 2
                self.prop("static.prop.coconutpalm" if palm else "static.prop.aporosatree",
                    carla.Transform(location,carla.Rotation(yaw=float((k*53+street.id*29)%360))),
                    "palm" if palm else "tree","vegetation",optional=True)

    # Street life. Buildings stand back by a forecourt strip beyond the
    # footpath (or the paved shoulder on village roads): stalls, chai tables,
    # crates, parked bicycles and people fill it, as in an Indian street. The
    # footpath itself stays free for walkers and nothing is ever placed on a
    # driving lane (align_and_check fails the build otherwise).
    LIFE = {
        "market_street":dict(forecourt=3.4,step=11.,mesh="apartment",scale=.62,boards=1,gap=(.8,2.6),
            items=("foodcart","chai","crates","barrels","garbage","chai","crates","foodcart","pots","people")),
        "divided_main_road":dict(forecourt=3.,step=17.,mesh="apartment",scale=.62,boards=2,gap=(3.,8.),
            items=("foodcart","bench","trashcan","pots","people","chai","crates")),
        "irregular_junction":dict(forecourt=2.8,step=16.,mesh="house",scale=.55,boards=3,gap=(2.,6.),
            items=("foodcart","barrels","roadwork","garbage","crates","people","chai")),
        "residential_colony":dict(forecourt=2.6,step=19.,mesh="house",scale=.6,boards=0,gap=(3.,8.),
            items=("pots","bench","clothesline","pots","wateringcan","people","clothesline")),
        "rural_road":dict(forecourt=3.,step=45.,mesh="house",scale=.5,boards=0,gap=(7.,16.),
            items=("hay","barrels","dhaba","debris","hay","wateringcan")),
    }

    # Mixed CARLA building meshes per district, each scaled from its measured
    # native height (metres), so rows read as varied concrete shop-houses and
    # flat-roofed homes rather than one repeated model.
    STATIC = "/Game/Carla/Static/"
    BUILDINGS = {
        "market_street":(("Building/SM_Apartment01_1.SM_Apartment01_1",7.5),("Building/SM_Apartment02_1.SM_Apartment02_1",8.5),
            ("Building/SM_TerracedHouse_1.SM_TerracedHouse_1",7.),("Building/SM_TerracedHouse02_1.SM_TerracedHouse02_1",7.5)),
        "divided_main_road":(("Building/SM_Apartment02_1.SM_Apartment02_1",10.),("Building/SM_Apartment01_1.SM_Apartment01_1",9.),
            ("Building/SM_TerracedHouse02_1.SM_TerracedHouse02_1",8.)),
        "irregular_junction":(("Building/SM_House04.SM_House04",6.5),("Building/SM_House10.SM_House10",6.5),
            ("Building/SM_TerracedHouse_1.SM_TerracedHouse_1",7.),("Building/SM_House11.SM_House11",6.)),
        "residential_colony":(("Building/SM_House01.SM_House01",6.),("Building/SM_House02.SM_House02",6.),
            ("Building/SM_House03.SM_House03",6.),("Building/SM_House05.SM_House05",6.5),("Building/SM_House06.SM_House06",6.)),
        "rural_road":(("Building/SM_House12_1.SM_House12_1",4.5),("Building/SM_House_13.SM_House_13",4.5),
            ("Building/SM_House03.SM_House03",4.8)),
    }
    BUILDING_GAP = {"market_street":(.2,1.2),"divided_main_road":(1.,3.),"irregular_junction":(1.,4.),
        "residential_colony":(3.,7.),"rural_road":(14.,30.)}
    POLE = "Pole/PoweLine/SM_ElectricPole01.SM_ElectricPole01"
    WIRE = "Other/SM_wire.SM_wire"
    AWNINGS = ("Building/Awning/SM_Awning01.SM_Awning01","Building/Awning/SM_Awning02.SM_Awning02")
    ACS = ("Building/SM_AirConditioner_1.SM_AirConditioner_1","Building/SM_AirConditioner_2.SM_AirConditioner_2")

    def measure(self,path):
        """Native mesh bounds at scale 1, measured once: extent and centre offset."""
        if path not in self.sizes:
            bp = self.world.get_blueprint_library().find("static.prop.mesh")
            bp.set_attribute("mesh_path",path)
            bp.set_attribute("scale","1")
            bp.set_attribute("mass","0")
            try:
                actor = self.world.spawn_actor(bp,carla.Transform(carla.Location(x=-900.,y=-900.,z=-400.)))
            except RuntimeError:
                self.sizes[path] = None
                return None
            self.world.tick(10)
            b = actor.bounding_box
            self.sizes[path] = (b.extent.x,b.extent.y,b.extent.z,b.location.x,b.location.y,b.location.z)
            actor.destroy()
        return self.sizes[path]

    def _static(self,path,location,yaw,scale,kind,project_class,base_z=None,centre=False,pitch=0.):
        """Spawn a measured mesh; base_z puts its lowest point there, centre puts
        its bounding-box centre at location (otherwise the pivot)."""
        size = self.measure(path)
        if size is None:
            return None
        ex,ey,ez,cx,cy,cz = size
        location = carla.Location(location.x,location.y,location.z)
        if centre:
            c,s_ = math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
            location.x -= (cx*c-cy*s_)*scale
            location.y -= (cx*s_+cy*c)*scale
            location.z -= cz*scale
        elif base_z is not None:
            location.z = base_z-(cz-ez)*scale
        bp = self.world.get_blueprint_library().find("static.prop.mesh")
        bp.set_attribute("role_name","indian_world_"+kind)
        bp.set_attribute("mesh_path",path)
        bp.set_attribute("scale",str(scale))
        bp.set_attribute("mass","0")
        actor = self.world.try_spawn_actor(bp,carla.Transform(location,carla.Rotation(yaw=yaw,pitch=pitch)))
        if actor is None:
            return None
        self._record(actor,kind,project_class,kind != "power_wire",mesh=path,scale=scale)
        return actor

    # Buildings keep close to their native size, so doors and windows stay
    # person-sized: shrinking a house to half size made it look half buried.
    SCALE_RANGE = (.85,1.1)
    MAX_WIDTH = 18.

    def fitting_buildings(self,district):
        """(path, scale, width, native half-height) for meshes that fit the
        district at near-native scale and at most MAX_WIDTH metres wide."""
        if district not in self.fitting:
            rows = []
            for path,height in self.BUILDINGS[district]:
                size = self.measure(self.STATIC+path)
                if size is None:
                    continue
                ex,ey,ez = size[:3]
                scale = float(np.clip(height/(2*ez),*self.SCALE_RANGE))
                width = 2*max(ex,ey)*scale
                if width <= self.MAX_WIDTH:
                    rows.append((path,scale,width,ez))
            self.fitting[district] = rows
        return self.fitting[district]

    def buildings(self,street,spec,lane,sign,forecourt,rng):
        """A continuous row of mixed buildings; returns their (centre s, width, height)."""
        life,length,placed = self.LIFE[street.district],spec["length"],[]
        # Built up right to the junction corners, as Indian junctions are;
        # align_and_check pushes back or drops anything reaching a lane there.
        s = 12.+4.*(sign < 0)
        k = 0
        while s < length-14.:
            if length > 115 and sign > 0 and abs(s-length*.53) < 9.:
                s += 4.
                continue
            fitting = self.fitting_buildings(street.district)
            if not fitting:
                break
            path,scale,width,ez = fitting[rng.randrange(len(fitting))]
            centre = s+width/2
            if centre+width/2 > length-12.:
                break
            wp = self.map.get_waypoint_xodr(street.id,lane,float(centre))
            outer = wp.lane_width/2+street.sidewalk+street.shoulder
            try:
                self.mesh(self.STATIC+path,carla.Transform(self.outside(wp,outer+forecourt+width),
                    carla.Rotation(yaw=wp.transform.rotation.yaw-90)),scale,"facade","building")
            except RuntimeError:
                s += 4.
                continue
            self.records[-1]["align_to"] = {"road":street.id,"lane":lane,"s":float(centre),
                "min_setback_from_driving_m":street.sidewalk+street.shoulder+forecourt}
            placed.append((centre,width,2*ez*scale,k))
            k += 1
            s += width+rng.uniform(*self.BUILDING_GAP[street.district])
        return placed

    def shopfronts(self,street,lane,forecourt,placed,rng):
        """Boards, awnings and window air-conditioners on the building faces."""
        life = self.LIFE[street.district]
        for centre,width,height,k in placed:
            wp = self.map.get_waypoint_xodr(street.id,lane,float(centre))
            face = wp.lane_width/2+street.sidewalk+street.shoulder+forecourt
            yaw = wp.transform.rotation.yaw
            if life["boards"] and k%life["boards"] == 0:
                board = self.board(wp,"shop_"+str(self.board_index%6),face+.25,shop=True)
                self.board_wishes.append((board,street,lane,float(centre)))
                self.board_index += 1
            if street.district != "rural_road" and height > 5.:
                for n in range(rng.randint(1,2)):
                    path = self.STATIC+self.ACS[rng.randrange(2)]
                    size = self.measure(path)
                    if size is None:
                        continue
                    # Mounted after alignment, on the wall a ray actually hits
                    # (a building's box face is often a porch or roof edge).
                    self.ac_wishes.append((street,lane,float(centre+rng.uniform(-width*.3,width*.3)),path,
                        rng.uniform(2.6,min(height-1.4,5.2))))

    def _wall_depth(self,wp,start,along_offsets,heights,boxes,avoid=()):
        """Distance beyond `start` (outward from the lane) to one flat building
        face, from horizontal rays; None if any ray misses or the face is uneven."""
        yaw = math.radians(wp.transform.rotation.yaw)
        out,along = (math.sin(yaw),-math.cos(yaw)),(math.cos(yaw),math.sin(yaw))
        inside = lambda q,bs:any(math.hypot(q.x-c.x,q.y-c.y) < r and box.contains(q,t) for box,t,c,r in bs)
        depths = []
        for da in along_offsets:
            for z in heights:
                a = self.outside(wp,start)
                a = carla.Location(a.x+along[0]*da,a.y+along[1]*da,z)
                b = carla.Location(a.x+out[0]*30.,a.y+out[1]*30.,z)
                hit = next((p for p in self.world.cast_ray(a,b) if inside(p.location,boxes) and not inside(p.location,avoid)),None)
                if hit is None:
                    return None
                depths.append((hit.location.x-a.x)*out[0]+(hit.location.y-a.y)*out[1])
        return min(depths) if max(depths)-min(depths) <= .12 else None

    @staticmethod
    def _grown_box(actor,grow):
        """Oriented world box of an actor, grown by `grow` metres, for point tests."""
        b,t = actor.bounding_box,actor.get_transform()
        box = carla.BoundingBox(b.location,carla.Vector3D(b.extent.x+grow,b.extent.y+grow,b.extent.z+grow))
        box.rotation = b.rotation
        centre = t.transform(b.location)
        return box,t,centre,math.hypot(b.extent.x,b.extent.y)*1.05+grow+1.

    def mount_boards(self,boxes):
        """Shop boards stand flush against the shop's real front wall (the box
        face is often a porch or roof edge); a board with no wall is removed."""
        self.boards_mounted,self.boards_removed,placed = 0,0,[]
        for actor,street,lane,s in self.board_wishes:
            if actor is None:
                continue
            wp = self.map.get_waypoint_xodr(street.id,lane,s)
            t = actor.get_transform()
            vertices = actor.bounding_box.get_world_vertices(t)
            yaw = math.radians(wp.transform.rotation.yaw)
            out,along = (math.sin(yaw),-math.cos(yaw)),(math.cos(yaw),math.sin(yaw))
            o = wp.transform.location
            lat = [(v.x-o.x)*out[0]+(v.y-o.y)*out[1] for v in vertices]
            alo = [(v.x-o.x)*along[0]+(v.y-o.y)*along[1] for v in vertices]
            zs = [v.z for v in vertices]
            mid,half = (min(alo)+max(alo))/2,(max(alo)-min(alo))/2
            start = wp.lane_width/2+street.sidewalk+street.shoulder+.5
            depth = self._wall_depth(wp,start,(mid-half*.8,mid,mid+half*.8),((min(zs)+max(zs))/2,max(zs)-.1),boxes)
            row = next(r for r in self.records if r.get("actor_id") == actor.id)
            if depth is None:
                actor.destroy()
                self.owned.remove(actor)
                del row["actor_id"]
                for name in row.get("object_names",()):
                    self.texture_groups[row["texture"]].remove(name)
                row["removed"] = "no flat shop wall behind the board"
                self.boards_removed += 1
                continue
            # The banner's box is far deeper than its thin panel, which passes
            # through the pivot: put the pivot just in front of the wall.
            # Respawned there (set_transform does not move a spawned mesh).
            pivot = (t.location.x-o.x)*out[0]+(t.location.y-o.y)*out[1]
            actor.destroy()
            self.owned.remove(actor)
            del row["actor_id"]
            for name in row.get("object_names",()):
                self.texture_groups[row["texture"]].remove(name)
            row["replaced_by_wall_mounted_board"] = True
            board = self.board(wp,row["texture"][len("banner_"):-len(".png")],start+depth-.12,shop=True)
            self.records[-1]["mounted_on_wall_shift_m"] = round(start+depth-.12-pivot,2)
            self.boards_mounted += 1
            placed.append(board)
        self.world.tick(3)
        return [self._grown_box(a,.05) for a in placed]

    def mount_acs(self):
        """Window ACs flush on a real flat wall; skipped where no wall is found."""
        self.ac_mounted,self.ac_skipped = 0,0
        # Spawned meshes carry no semantic tag, so a hit counts as wall only
        # inside a building's world box (not a pole, board or tree).
        by_id = {a.id:a for a in self.owned}
        boxes = [self._grown_box(by_id[r["actor_id"]],.15) for r in self.records if r.get("kind") == "facade" and "actor_id" in r]
        board_boxes = self.mount_boards(boxes)
        for street,lane,s,path,z in self.ac_wishes:
            size = self.measure(path)
            wp = self.map.get_waypoint_xodr(street.id,lane,s)
            if size is None or wp is None:
                continue
            ex,ey,ez = size[:3]
            half_w,half_d = max(ex,ey),min(ex,ey)
            start = wp.lane_width/2+street.sidewalk+street.shoulder+.5
            depth = self._wall_depth(wp,start,(-half_w,0.,half_w),(z,z+ez,z+2*ez),boxes,board_boxes)
            if depth is None:
                self.ac_skipped += 1
                continue
            spot = self.outside(wp,start+depth-half_d-.02)
            spot.z = z+ez
            turn = 90. if ex < ey else 0.
            if self._static(path,spot,wp.transform.rotation.yaw+turn,1.,"window_ac","building",centre=True) is not None:
                self.ac_mounted += 1

    def power_line(self,street,spec,lane,forecourt,avoid):
        """Electric poles on one side with overhead wires between them."""
        pole = self.STATIC+self.POLE
        size = self.measure(pole)
        if size is None:
            return
        scale = 8.5/(2*size[2])
        tops = []
        s = 18.
        while s < spec["length"]-18.:
            if any(abs(s-c) < r for c,r in avoid):
                s += 3.
                continue
            wp = self.map.get_waypoint_xodr(street.id,lane,float(s))
            outer = wp.lane_width/2+street.sidewalk+street.shoulder
            location = self.outside(wp,outer+(.45 if street.sidewalk else .35)*forecourt)
            if not self._on_drive(location.x,location.y) and \
                    self._static(pole,location,wp.transform.rotation.yaw,scale,"electric_pole","pole",base_z=0.) is not None:
                tops.append((location.x,location.y,8.5-.6))
            else:
                tops.append(None)
            s += 30.
        wire = self.STATIC+self.WIRE
        size = self.measure(wire)
        if size is None:
            self.wire_note = "SM_wire not available"
            return
        ex,ey,ez = size[:3]
        long,thin = max(ex,ey),max(min(ex,ey),ez)
        if long/max(thin,1e-3) < 12.:
            self.wire_note = "SM_wire is not a straight thin cable: "+str(size[:3])
            return
        for a,b in zip(tops,tops[1:]):
            if a is None or b is None:
                continue
            span = math.dist(a[:2],b[:2])
            scale = span/(2*long)
            if 2*thin*scale > .14:
                self.wire_note = "SM_wire too thick at span scale"
                continue
            heading = math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))+(0. if ex >= ey else 90.)
            nx,ny = -math.sin(math.radians(heading)),math.cos(math.radians(heading))
            for side in (-.35,.35):
                mid = carla.Location((a[0]+b[0])/2+nx*side,(a[1]+b[1])/2+ny*side,a[2])
                self._static(wire,mid,heading,scale,"power_wire","pole",centre=True)

    def _respawn(self,actor,row,transform):
        """Move a spawned static mesh. set_transform updates the actor record
        but not the rendered mesh or its collision (buildings stayed 4-9 m
        behind their aligned place), so spawn it again at the new pose."""
        bp = self.world.get_blueprint_library().find("static.prop.mesh")
        bp.set_attribute("role_name","indian_world_"+row["kind"])
        bp.set_attribute("mesh_path",row["mesh"])
        bp.set_attribute("scale",str(row["scale"]))
        bp.set_attribute("mass","0")
        new = self.world.spawn_actor(bp,transform)
        new.set_collisions(row["collisions_enabled"])
        actor.destroy()
        self.owned[self.owned.index(actor)] = new
        row["actor_id"] = new.id
        return new

    def align_and_check(self):
        """Settle each building before the shared alignment and lane check: on a
        bend a wide building can reach a lane with its front at the setback.
        Move it back a metre at a time; drop it if 6 m more is not enough."""
        actor_by_id = {a.id:a for a in self.owned}
        self.dropped_buildings = 0
        for row in self.records:
            if "align_to" not in row:
                continue
            actor,spec = actor_by_id[row["actor_id"]],row["align_to"]
            wp = self.map.get_waypoint_xodr(spec["road"],spec["lane"],spec["s"])
            t,box = actor.get_transform(),actor.bounding_box
            vertices = box.get_world_vertices(t)
            forward = wp.transform.get_forward_vector()
            yaw = math.radians(wp.transform.rotation.yaw)
            outside = (math.sin(yaw),-math.cos(yaw))
            origin = wp.transform.location
            along = [(v.x-origin.x)*forward.x+(v.y-origin.y)*forward.y for v in vertices]
            lateral = [(v.x-origin.x)*outside[0]+(v.y-origin.y)*outside[1] for v in vertices]
            samples = [t.transform(carla.Location(x=float(x)+box.location.x,y=float(y)+box.location.y,z=box.location.z))
                for x in np.linspace(-box.extent.x,box.extent.x,5) for y in np.linspace(-box.extent.y,box.extent.y,5)]
            centre = (min(along)+max(along))/2
            for extra in range(7):
                shift = wp.lane_width/2+spec["min_setback_from_driving_m"]+extra-min(lateral)
                dx,dy = shift*outside[0]-centre*forward.x,shift*outside[1]-centre*forward.y
                if not any(self._on_drive(q.x+dx,q.y+dy) for q in samples):
                    spec["min_setback_from_driving_m"] += extra
                    if extra:
                        spec["pushed_back_m"] = extra
                    break
            else:
                actor.destroy()
                self.owned.remove(actor)
                del row["actor_id"],row["align_to"]
                row["removed"] = "would reach a driving lane on this bend"
                self.dropped_buildings += 1
        # Same alignment as WorldScenery.align_and_check, then a final check in
        # which a building still touching a lane is removed rather than failing
        # the whole run; any other object touching a lane still fails it.
        for row in self.records:
            if "align_to" not in row:
                continue
            actor,spec = actor_by_id[row["actor_id"]],row["align_to"]
            wp = self.map.get_waypoint_xodr(spec["road"],spec["lane"],spec["s"])
            t = actor.get_transform()
            vertices = actor.bounding_box.get_world_vertices(t)
            forward = wp.transform.get_forward_vector()
            yaw = math.radians(wp.transform.rotation.yaw)
            outside = (math.sin(yaw),-math.cos(yaw))
            along = [(v.x-wp.transform.location.x)*forward.x+(v.y-wp.transform.location.y)*forward.y for v in vertices]
            lateral = [(v.x-wp.transform.location.x)*outside[0]+(v.y-wp.transform.location.y)*outside[1] for v in vertices]
            shift = wp.lane_width/2+spec["min_setback_from_driving_m"]-min(lateral)
            centre = (min(along)+max(along))/2
            t.location.x += shift*outside[0]-centre*forward.x
            t.location.y += shift*outside[1]-centre*forward.y
            # Height stays at the spawn pivot (ground level). A mesh's box can
            # reach below its floor or start above it, so lifting the box
            # bottom to the ground left houses floating or half sunk.
            actor = self._respawn(actor,row,t)
            actor_by_id[actor.id] = actor
        self.world.tick(15)
        failures,removed = [],False
        for row in self.records:
            if row.get("kind") in ("ground","asphalt","median") or "actor_id" not in row:
                continue
            actor = actor_by_id[row["actor_id"]]
            box,t = actor.bounding_box,actor.get_transform()
            vertices = box.get_world_vertices(t)
            row["world_bounds_xyz"] = [[min(getattr(v,a) for v in vertices),max(getattr(v,a) for v in vertices)] for a in ("x","y","z")]
            hits,samples = [],0
            for x in np.linspace(-box.extent.x,box.extent.x,5):
                for y in np.linspace(-box.extent.y,box.extent.y,5):
                    p = t.transform(carla.Location(x=float(x)+box.location.x,y=float(y)+box.location.y,z=box.location.z))
                    samples += 1  # height does not matter: the lane lookup is 2D
                    if self._on_drive(p.x,p.y):
                        hits.append([p.x,p.y])
            row["driving_lane_overlap_samples_checked"] = samples
            if hits and row.get("kind") == "facade":
                actor.destroy()
                self.owned.remove(actor)
                del row["actor_id"]
                row.pop("align_to",None)
                row["removed"] = "reached a driving lane after alignment"
                self.dropped_buildings += 1
                removed = True
            elif hits:
                failures.extend({"id":actor.id,"kind":row["kind"],"xy":xy} for xy in hits)
        if failures:
            raise ValueError("Scenery intrudes into driving geometry: "+str(failures[:5]))
        if removed:
            self.world.tick(5)

    def _put(self,street,lane,s,depth,identifier,kind,project_class="other",turn=0.):
        """One optional prop in the forecourt, depth metres beyond footpath/shoulder."""
        wp = self.map.get_waypoint_xodr(street.id,lane,float(s))
        if wp is None:
            return None
        outer = wp.lane_width/2+street.sidewalk+street.shoulder
        location = self.outside(wp,outer+depth)
        if self._on_drive(location.x,location.y):
            return None
        return self.prop(identifier,carla.Transform(location,carla.Rotation(yaw=wp.transform.rotation.yaw+turn)),
            kind,project_class,optional=True)

    STATIONARY = ("vehicle.vespa.zx125","vehicle.vespa.zx125","vehicle.yamaha.yzf","vehicle.bh.crossbike","vehicle.gazelle.omafiets")

    def _parked(self,street,lane,s,depth,identifier,rng):
        wp = self.map.get_waypoint_xodr(street.id,lane,float(s))
        if wp is None or self.parked >= 30:
            return
        location = self.outside(wp,wp.lane_width/2+street.sidewalk+street.shoulder+depth)
        if self._on_drive(location.x,location.y):
            return
        location.z = .08
        bp = self.world.get_blueprint_library().find(identifier)
        bp.set_attribute("role_name","indian_world_stationary_two_wheeler")
        if bp.has_attribute("color"):
            values = bp.get_attribute("color").recommended_values
            if values:
                bp.set_attribute("color",rng.choice(values))
        actor = self.world.try_spawn_actor(bp,carla.Transform(location,carla.Rotation(yaw=wp.transform.rotation.yaw+rng.choice((55.,65.,-120.)))))
        if actor is not None:
            actor.set_simulate_physics(False)
            self._record(actor,"stationary_two_wheeler","bicycle" if "bike" in identifier or "omafiets" in identifier else "motorcycle",
                True,blueprint=identifier,physics=False)
            self.parked += 1

    def _person(self,street,lane,s,depth,rng):
        wp = self.map.get_waypoint_xodr(street.id,lane,float(s))
        if wp is None or self.bystanders >= 18:
            return
        location = self.outside(wp,wp.lane_width/2+street.sidewalk+street.shoulder+depth)
        location.z = 1.
        bp = self.walker_bps[rng.randrange(len(self.walker_bps))]
        actor = self.world.try_spawn_actor(bp,carla.Transform(location,carla.Rotation(yaw=wp.transform.rotation.yaw+rng.choice((80.,-100.,0.,180.)))))
        if actor is not None:
            self._record(actor,"bystander","person",True,blueprint=bp.id)
            self.bystanders += 1

    def _item(self,street,lane,s,name,depth,rng):
        """Place one street-life group starting at s; returns its length along the road."""
        put = lambda ds,d,identifier,kind,cls="other",turn=0.:self._put(street,lane,s+ds,depth*d,identifier,kind,cls,turn)
        if name == "kiosk":
            put(1.2,.5,"static.prop.kiosk_01","kiosk","building")
            return 2.6
        if name == "foodcart":
            put(2.4,.5,"static.prop.foodcart","food_cart",turn=90.)
            return 5.
        if name in ("chai","dhaba"):
            put(.9,.55,"static.prop.plastictable","chai_table")
            put(.0,.55,"static.prop.plasticchair","chai_chair",turn=90.)
            put(1.8,.55,"static.prop.plasticchair","chai_chair",turn=-90.)
            if name == "dhaba":
                put(4.6,.5,"static.prop.foodcart","food_cart",turn=90.)
            if rng.random() < .7:
                self._person(street,lane,s+.9,depth*.4,rng)
            return 7.2 if name == "dhaba" else 2.6
        if name == "crates":
            for k,identifier in enumerate(("box01","box02","creasedbox01","box03")[:rng.randint(2,4)]):
                put(.1+.7*k,.75,"static.prop."+identifier,"crates")
            return 2.8
        if name == "bikes":
            # CARLA two-wheelers always carry a rider, so these are riders
            # waiting on stationary scooters and bicycles, physics off.
            count = rng.randint(1,3)
            for k in range(count):
                self._parked(street,lane,s+.6+1.2*k,depth*.42,rng.choice(self.STATIONARY),rng)
            return count*1.2+.8
        if name == "barrels":
            for k in range(rng.randint(2,3)):
                put(.4+.75*k,.7,"static.prop.barrel","barrel")
            return 2.2
        if name == "garbage":
            put(.4,.3,"static.prop.garbage0"+str(rng.randint(1,6)),"garbage")
            put(1.2,.35,"static.prop.trashbag","garbage")
            return 1.8
        if name in ("vending","atm","advert"):
            identifier = {"vending":"vendingmachine","atm":"atm","advert":"advertisement"}[name]
            put(.8,.75 if name != "advert" else .55,"static.prop."+identifier,name,turn=-90.)
            return 2.6 if name == "advert" else 1.6
        if name == "bench":
            put(1.,.5,"static.prop.bench0"+str(rng.randint(1,3)),"bench")
            if rng.random() < .5:
                self._person(street,lane,s+1.,depth*.4,rng)
            return 2.2
        if name == "trashcan":
            put(.4,.3,"static.prop.trashcan0"+str(rng.randint(1,5)),"trashcan")
            return 1.
        if name == "pots":
            for k in range(rng.randint(2,3)):
                put(.4+.8*k,.75,"static.prop.plantpot0"+str(rng.randint(1,8)),"plant_pot")
            return 2.4
        if name == "doghouse":
            put(.8,.6,"static.prop.doghouse","doghouse")
            return 1.6
        if name == "mailbox":
            put(.4,.25,"static.prop.mailbox","mailbox")
            return .8
        if name == "clothesline":
            put(1.5,.6,"static.prop.clothesline","clothesline")
            return 3.2
        if name == "lamp":
            put(.3,.15,"static.prop.gardenlamp","lamp")
            return .6
        if name == "roadwork":
            put(.5,.5,"static.prop.warningconstruction","roadwork")
            put(1.8,.45,"static.prop.constructioncone","roadwork")
            put(2.6,.45,"static.prop.constructioncone","roadwork")
            return 3.2
        if name == "hay":
            put(.8,.55,"static.prop.haybale","hay")
            if rng.random() < .6:
                put(2.4,.6,"static.prop.haybalelb","hay")
            return 3.4
        if name == "debris":
            put(.5,.5,"static.prop.dirtdebris0"+str(rng.randint(1,3)),"debris")
            put(1.5,.6,"static.prop.brokentile0"+str(rng.randint(1,4)),"debris")
            return 2.2
        if name == "wateringcan":
            put(.4,.6,"static.prop.wateringcan","household")
            put(1.,.7,"static.prop.plantpot0"+str(rng.randint(1,8)),"plant_pot")
            return 1.5
        if name == "people":
            for k in range(rng.randint(1,2)):
                self._person(street,lane,s+.6+.9*k,depth*(.35+.2*k),rng)
            return 2.
        raise ValueError(name)

    def street_life(self,street,spec,lane,sign,avoid):
        life = self.LIFE[street.district]
        rng = random.Random(street.id*97+(sign > 0))
        s = 28.+rng.uniform(0.,4.)
        while s < spec["length"]-30.:
            if any(abs(s-centre) < radius for centre,radius in avoid):
                s += 2.
                continue
            name = life["items"][rng.randrange(len(life["items"]))]
            s += self._item(street,lane,s,name,life["forecourt"],rng)+rng.uniform(*life["gap"])

    def build(self,progress=lambda stage:None):
        self.walker_bps = sorted(self.world.get_blueprint_library().filter("walker.pedestrian.*"),key=lambda bp:bp.id)
        self.bystanders,self.parked = 0,0
        self.sizes,self.board_index,self.awning_turn,self.wire_note = {},0,0.,None
        self.ac_wishes,self.board_wishes,self.fitting = [],[],{}
        progress("placing_ground")
        before = set(self.world.get_names_of_all_objects())
        self.mesh(MESH,carla.Transform(carla.Location(x=60,y=-60,z=-.08-.25*75)),75.,"ground","terrain",collision=False)
        names = sorted(set(self.world.get_names_of_all_objects())-before)
        self.world.apply_color_texture_to_objects(names,carla.MaterialParameter.Diffuse,asphalt_texture((56,47,35),20261024))
        self.mesh("/Engine/BasicShapes/Plane.Plane",carla.Transform(carla.Location(x=60,y=-60,z=-.12)),
            1800.,"ground","terrain",collision=True)
        progress("placing_connected_road_surfaces")
        surface = self.surfaces()
        index = 0
        for spec in self.network["streets"]:
            street = BY_ID[spec["id"]]
            self.current_street = street
            self.profile = SimpleNamespace(sidewalk=street.sidewalk,shoulder=street.shoulder)
            progress("placing_"+street.district+"_street_"+str(street.id))
            life = self.LIFE[street.district]
            forecourt = life["forecourt"]
            length = spec["length"]
            crossings = jaywalk_spots(street,length)
            for sign in (1,-1):
                lane = sign*street.lanes[-1]
                # Keep the bus stop, signs and jaywalker crossing places clear.
                avoid = [(55. if sign > 0 else length-55.,3.5)]+[(c,4.) for c in crossings]
                if length > 115 and sign > 0:
                    avoid += [(length*.53,7.),(length*.72,3.5)]
                rng = random.Random(street.id*31+(sign > 0))
                placed = self.buildings(street,spec,lane,sign,forecourt,rng)
                self.shopfronts(street,lane,forecourt,placed,rng)
                progress("placing_"+street.district+"_street_life_"+str(street.id))
                if sign > 0:
                    # Poles first, so forecourt props never spawn into them.
                    self.power_line(street,spec,lane,forecourt,avoid)
                self.street_life(street,spec,lane,sign,avoid)
                self.greenery(street,spec,lane)
                if spec["length"] > 115:
                    wp = self.map.get_waypoint_xodr(street.id,lane,55. if sign > 0 else spec["length"]-55.)
                    outer = wp.lane_width/2+street.sidewalk+street.shoulder
                    self.board(wp,"speed_"+str(40 if street.divided else 30 if street.shoulder else 20),outer+1.1)
                    if sign > 0:
                        stop = self.map.get_waypoint_xodr(street.id,lane,spec["length"]*.53)
                        self.prop("static.prop.busstop",carla.Transform(self.outside(stop,outer+3.3),
                            carla.Rotation(yaw=stop.transform.rotation.yaw)),"bus_stop","building")
                        self.board(stop,"bus_stop",outer+1.1)
                        if street.district in ("residential_colony","rural_road"):
                            label = "school" if street.sidewalk else "village"
                            wp = self.map.get_waypoint_xodr(street.id,lane,spec["length"]*.72)
                            self.board(wp,label,outer+1.1)
            if street.divided:
                before = set(self.world.get_names_of_all_objects())
                for i,s in enumerate(np.arange(5.,spec["length"]-4.,4.8)):
                    for side in (1,-1):
                        wp = self.map.get_waypoint_xodr(street.id,side,float(s))
                        self.mesh(MESH,carla.Transform(carla.Location(x=wp.transform.location.x,y=wp.transform.location.y,z=.17-.25*.052),
                            carla.Rotation(yaw=wp.transform.rotation.yaw)),.052,"median","median",collision=False)
                objects = sorted(set(self.world.get_names_of_all_objects())-before)
                self.world.apply_color_texture_to_objects(objects,carla.MaterialParameter.Diffuse,asphalt_texture((65,55,25),17))
        self.current_street = None
        progress("checking_all_native_scenery_bounds")
        self.world.tick(20)
        self.align_and_check()
        progress("mounting_window_acs_on_walls")
        self.mount_acs()
        self.world.tick(5)
        progress("applying_indian_sign_artwork")
        for name,objects in self.texture_groups.items():
            self.world.apply_color_texture_to_objects(objects,carla.MaterialParameter.Diffuse,image_texture(ASSETS/"textures"/name))
        actor_by_id = {a.id:a for a in self.owned}
        for row in self.records:
            if "actor_id" not in row:
                continue
            actor = actor_by_id[row["actor_id"]]
            t = actor.get_transform()
            row["native_semantic_tags"] = list(actor.semantic_tags)
            row["project_label_requires_instance_override"] = row["kind"] in ("asphalt","ground","median","shop_board","indian_sign")
            row["transform"] = {"xyz":[t.location.x,t.location.y,t.location.z],
                "pitch_yaw_roll":[t.rotation.pitch,t.rotation.yaw,t.rotation.roll]}
        return {"schema":"combined_india_scenery_v1","single_world":True,"objects":self.records,"surface":surface,
            "raw_semantics_are_training_ready":False,"label_policy":"Native evidence unchanged; exact owned instance corrections required",
            "districts_present":sorted({row.get("district") for row in self.records if row.get("district")}),
            "authenticity":"Original Hindi/English artwork; generic installed vehicle/building meshes; no live autorickshaw",
            "measured_meshes":{path:size for path,size in self.sizes.items()},"power_wire_note":self.wire_note,
            "buildings_dropped_on_bends":self.dropped_buildings,
            "window_acs":{"mounted_on_wall":self.ac_mounted,"skipped_no_flat_wall":self.ac_skipped},
            "shop_boards":{"mounted_on_wall":self.boards_mounted,"removed_no_wall":self.boards_removed}}
