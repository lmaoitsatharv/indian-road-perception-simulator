"""Create original Indian shop/sign textures and an import-ready three-wheel asset.

The OBJ is source artwork, NOT a registered CARLA physics vehicle. Its manifest
deliberately blocks runtime/training use until cooked mesh, collision and semantic
tests exist. No downloaded models, borrowed brand names or external licences.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "integration/indian_world_assets"
SHOPS = [("किराना स्टोर", "KIRANA STORE", "GROCERIES  •  DAILY NEEDS", (30, 84, 62)),
         ("चाय और नाश्ता", "CHAI & NASHTA", "FRESH TEA  •  SNACKS", (140, 51, 31)),
         ("फल और सब्ज़ी", "FRESH SABZI", "FRUIT  •  VEGETABLES", (46, 91, 33)),
         ("दवा की दुकान", "MEDICAL STORE", "PHARMACY", (32, 76, 112)),
         ("साइकिल मरम्मत", "CYCLE REPAIR", "CYCLE  •  PUNCTURE", (102, 65, 28)),
         ("दूध और डेयरी", "MILK & DAIRY", "MILK  •  CURD", (34, 88, 120))]


def font(size):
    for path in (Path("C:/Windows/Fonts/Nirmala.ttc"), Path("C:/Windows/Fonts/mangal.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError("A Unicode font is needed for the original shop/sign textures")


def text(draw, xy, value, size, color):
    draw.text(xy, value, font=font(size), fill=color, anchor="mm")


def textures():
    folder = TARGET / "textures"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (hindi, english, sub, color) in enumerate(SHOPS):
        image = Image.new("RGB", (768, 320), color)
        d = ImageDraw.Draw(image)
        d.rectangle((9, 9, 758, 310), outline=(239, 215, 144), width=5)
        text(d, (384, 67), hindi, 49, (255, 242, 200))
        text(d, (384, 171), english, 60, "white")
        text(d, (384, 259), sub, 26, (255, 226, 171))
        name = "shop_"+str(i)+".png"
        image.save(folder/name)
        records.append({"file": name, "kind": "shopfront", "text": [hindi, english, sub], "background":color})
    for speed in (20, 30, 40):
        image = Image.new("RGB", (512, 512), (236, 233, 219))
        d = ImageDraw.Draw(image)
        d.ellipse((38, 38, 474, 474), fill="white", outline=(169, 21, 24), width=33)
        text(d, (256, 241), str(speed), 195, (22, 24, 25))
        text(d, (256, 380), "km/h", 32, (22, 24, 25))
        name = f"speed_{speed}.png"
        image.save(folder/name)
        records.append({"file": name, "kind": "traffic_sign", "text": [str(speed), "km/h"]})
    for name, hindi, english, color in (
        ("bus_stop", "बस स्टॉप", "BUS STOP", (27, 70, 111)),
        ("keep_left", "बाईं ओर चलें", "KEEP LEFT", (28, 82, 54)),
        ("village", "गाँव की सड़क", "VILLAGE ROAD", (34, 92, 62)),
        ("school", "विद्यालय क्षेत्र", "SCHOOL ZONE", (141, 69, 25)),
        ("delivery", "स्थानीय डिलीवरी", "LOCAL DELIVERY", (33, 86, 71))):
        image = Image.new("RGB", (768, 320), color)
        d = ImageDraw.Draw(image)
        d.rectangle((10, 10, 758, 310), outline=(238, 236, 218), width=6)
        text(d, (384, 96), hindi, 59, "white")
        text(d, (384, 223), english, 56, "white")
        image.save(folder/(name+".png"))
        records.append({"file": name+".png", "kind": "traffic_sign" if name != "delivery" else "vehicle_livery",
                        "text": [hindi, english], "background":color})
    # Native Windows text shaping handles Devanagari conjuncts; this Pillow
    # installation has no RAQM and must not silently draw unshaped Hindi.
    layout = folder/"text_layout.json"
    layout.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    subprocess.run(["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",
        str(ROOT/"setup/render_indian_sign_text.ps1"),"-Layout",str(layout)],check=True)
    for row in records:
        row["sha256"] = hashlib.sha256((folder/row["file"]).read_bytes()).hexdigest()
        # Native short banner's cloth consumes only part of its texture atlas.
        # A vertical atlas carries a sideways board for the rotated native mesh.
        art = Image.open(folder/row["file"]).convert("RGB")
        canvas = Image.new("RGB",(1024,1024),row.get("background",(236,233,219)))
        # Native UV-grid probe: visible cloth U=.239..610, V=.005..640.
        # Keep a small inset so no lettering sits on a clipped cloth edge.
        # Native roll/yaw orientation requires clockwise artwork. The opposite
        # rotation renders intact text upside down even though UV bounds fit.
        art = art.rotate(270,expand=True).resize((354,620),Image.Resampling.LANCZOS)
        canvas.paste(art,(250,15))
        canvas.save(folder/("banner_"+row["file"]))
        row["banner_file"] = "banner_"+row["file"]
        row["banner_sha256"] = hashlib.sha256((folder/row["banner_file"]).read_bytes()).hexdigest()
        row["native_uv_grid_probe_rect_uv"] = [.239,.005,.610,.640]
        row["artwork_rotation_clockwise_degrees"] = 90
    return records


class Mesh:
    def __init__(self):
        self.vertices, self.faces = [], []

    def box(self, center, size, material):
        cx, cy, cz = center
        a, b, c = (v/2 for v in size)
        start = len(self.vertices)+1
        self.vertices += [(cx+x, cy+y, cz+z) for x, y, z in
            ((-a,-b,-c),(a,-b,-c),(a,b,-c),(-a,b,-c),(-a,-b,c),(a,-b,c),(a,b,c),(-a,b,c))]
        self.faces += [(material, [start+i for i in face]) for face in
            ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))]

    def wheel(self, center, radius=.23, width=.13, material="rubber", segments=32):
        x, y, z = center
        start = len(self.vertices)+1
        for side in (-1, 1):
            self.vertices += [(x+radius*math.cos(2*math.pi*i/segments), y+side*width/2,
                               z+radius*math.sin(2*math.pi*i/segments)) for i in range(segments)]
        self.faces.append((material, list(range(start+segments-1, start-1, -1))))
        self.faces.append((material, list(range(start+segments, start+2*segments))))
        for i in range(segments):
            j = (i+1)%segments
            self.faces.append((material, [start+i,start+j,start+segments+j,start+segments+i]))


def autorickshaw():
    mesh = Mesh()
    # Metres, X forward / Y left / Z up. Original unbranded Indian-style form.
    mesh.box((-.20,0,.45), (2.12,1.27,.17), "green")
    mesh.box((-.67,0,.82), (.13,1.30,.70), "green")
    for y in (-.62,.62):
        mesh.box((-.56,y,.66), (1.48,.07,.42), "green")
        mesh.box((-.55,y,1.52), (1.46,.045,.63), "canvas")
        for x in (-1.05,.18):
            mesh.box((x,y,1.43), (.045,.045,.93), "metal")
    mesh.box((-.48,0,1.87), (1.72,1.36,.11), "yellow")
    mesh.box((.42,0,1.76), (.64,1.22,.11), "yellow")
    mesh.box((.82,0,1.40), (.06,1.13,.60), "glass")
    for y in (-.59,.59):
        mesh.box((.79,y,1.4), (.05,.05,.7), "yellow")
    mesh.box((.82,0,.96), (.12,1.22,.36), "green")
    mesh.box((1.03,0,.69), (.55,.66,.19), "green")
    mesh.box((-.75,0,1.05), (.38,1.09,.12), "seat")
    mesh.box((-.98,0,1.24), (.11,1.09,.34), "seat")
    mesh.box((.12,0,.93), (.38,.53,.13), "seat")
    mesh.box((-.05,0,1.12), (.10,.53,.30), "seat")
    mesh.box((.48,0,1.10), (.045,.045,.35), "metal")
    mesh.box((.48,0,1.27), (.045,.48,.045), "metal")
    wheels = [(.96,0,.25),(-.77,-.64,.25),(-.77,.64,.25)]
    for wheel in wheels:
        mesh.wheel(wheel)
        mesh.wheel(wheel, radius=.11, width=.141, material="metal")
    mesh.box((1.13,0,.91), (.025,.22,.17), "lamp")
    for y in (-.45,.45):
        mesh.box((-1.28,y,.69), (.035,.13,.15), "red")
    mesh.box((1.15,0,.53), (.025,.27,.11), "plate")
    mesh.box((-1.29,0,.53), (.025,.31,.12), "plate")
    folder = TARGET / "autorickshaw_source"
    folder.mkdir(exist_ok=True)
    lines = ["# Original unbranded three-wheel source; metres; X forward Y left Z up", "mtllib autorickshaw.mtl"]
    lines += ["v %.6f %.6f %.6f" % v for v in mesh.vertices]
    material = None
    for mat, face in mesh.faces:
        if mat != material:
            lines.append("usemtl "+mat)
            material = mat
        lines.append("f "+" ".join(map(str, face)))
    (folder/"autorickshaw.obj").write_text("\n".join(lines)+"\n")
    colors = {"green":(.035,.30,.12),"yellow":(.96,.65,.035),"canvas":(.055,.07,.06),
        "rubber":(.018,.020,.022),"metal":(.31,.33,.34),"seat":(.05,.04,.035),
        "glass":(.25,.42,.43),"lamp":(.94,.90,.68),"red":(.61,.035,.025),"plate":(.94,.70,.04)}
    (folder/"autorickshaw.mtl").write_text("\n".join(
        f"newmtl {name}\nKd {' '.join(map(str,color))}\nKs 0.12 0.12 0.12\nNs 20\n" for name,color in colors.items()))
    record = {"status":"source_artwork_only_not_registered", "units":"metres", "axis":"X forward, Y left, Z up",
        "wheel_centres_m":wheels,"wheel_radius_m":.23,"wheel_count":3,
        "intended_project_class":"autorickshaw", "vertices":len(mesh.vertices),"faces":len(mesh.faces),
        "collision_hull_proposal":{"center_m":[-.1,0,1.04],"size_m":[2.7,1.43,1.92]},
        "native_blueprint":None,"physics_verified":False,"native_semantics_verified":False,
        "runtime_allowed":False,"training_allowed":False,
        "required_next":["CARLA-compatible Unreal editor/source build and asset cooking",
            "Rig three wheels and validate native vehicle physics/collision envelope",
            "Register blueprint and verify semantic/instance pixels before training"],
        "licence":"Original project artwork; no third-party source model", "brand":"unbranded illustrative form"}
    record["obj_sha256"] = hashlib.sha256((folder/"autorickshaw.obj").read_bytes()).hexdigest()
    (folder/"asset_manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


if __name__ == "__main__":
    TARGET.mkdir(exist_ok=True)
    result = {"schema":"indian_world_assets_v1","textures":textures(),"autorickshaw":autorickshaw()}
    (TARGET/"manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"textures":len(result["textures"]),"autorickshaw":result["autorickshaw"]["status"]}))
