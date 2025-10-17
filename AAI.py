"""
Procedural Floor Plan Generator (AAI) — improved realistic details

Highlights:
- Variable room dimensions with small randomness (realistic proportions)
- Trilogy core (Living, Dining, Kitchen) with shared walls and doors
- Corridor/hall layout for bedroom zone
- Baths attached to bedrooms (configurable)
- Smarter non-overlap shifting with retries and axis-priority
- Doors placed on shared wall segments with offsets and clearances
- Windows on external walls
- Furniture placement with bounding checks and basic collision avoidance
- JSON export with full geometry + PNG render with walls, doors, windows, labels, furniture
- Integrates ArchitectAgent via --agent for natural-language goals
"""
import json
import math
import random as rnd
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------- CONFIG / CONSTANTS -----------------
FT = 12  # drawing units per foot
WALL_THICKNESS_FT = 0.6
DOOR_WIDTH_FT = 3.0
WINDOW_MIN_FT = 3.0

WALL_TH = WALL_THICKNESS_FT * FT
DOOR_W = DOOR_WIDTH_FT * FT
MIN_CLEAR_FT = 1.5  # min overlap for shared wall to allow a door

# Typical nominal sizes (w_ft, h_ft) — we sample +/- variability
ROOM_NOMINAL = {
    "Living Room": (16, 16),
    "Dining Room": (12, 10),
    "Kitchen": (10, 8),
    "Bedroom": (12, 12),
    "Bathroom": (8, 6),
    "Laundry": (6, 6),
    "Porch": (10, 6),
    "Storage": (6, 6),
    "Hall": (10, 6)
}

FURNITURE = {
    "Living Room": [("sofa", 6, 3), ("armchair", 3, 3), ("coffee_table", 4, 2), ("tv", 3, 1)],
    "Dining Room": [("dining_table", 6, 4)],
    "Kitchen": [("fridge", 3, 3), ("stove", 3, 3), ("sink", 3, 2)],
    "Bedroom": [("bed", 6, 6), ("nightstand", 2, 2), ("wardrobe", 4, 2)],
    "Bathroom": [("toilet", 2.5, 2), ("sink", 2.5, 2), ("bathtub", 5, 3)],
    "Laundry": [("washer", 3, 3), ("dryer", 3, 3)]
}

FURN_COLORS = {
    "bed": "purple", "nightstand": "gray", "wardrobe": "dimgray",
    "sofa": "saddlebrown", "armchair": "tan", "coffee_table": "sienna", "tv": "black",
    "dining_table": "peru", "sink": "lightblue", "stove": "orange", "fridge": "silver",
    "washer": "gray", "dryer": "gray", "toilet": "white", "bathtub": "lightblue"
}

# ----------------- UTILITIES -----------------
def ft(val): return val * FT

def wall_thickness(): return WALL_TH

def door_width(): return DOOR_W

Rect = namedtuple('Rect', ['x','y','w','h'])  # simple rectangle helper

def rect_to_dict(r, rtype=None, name=None):
    return {
        'x': r.x,
        'y': r.y,
        'w': r.w,
        'h': r.h,
        'type': rtype or 'room',
        'name': name or rtype or 'room'
    }

def rect_overlap(a: Rect, b: Rect, eps=1e-6):
    return not (a.x + a.w <= b.x + eps or b.x + b.w <= a.x + eps or a.y + a.h <= b.y + eps or b.y + b.h <= a.y + eps)

def area(r: Rect): return r.w * r.h

def sample_room_dims(kind, jitter=0.12):
    if kind not in ROOM_NOMINAL:
        w_ft, h_ft = ROOM_NOMINAL.get('Storage', (6,6))
    else:
        w_ft, h_ft = ROOM_NOMINAL[kind]
    factor_w = 1.0 + rnd.uniform(-jitter, jitter)
    factor_h = 1.0 + rnd.uniform(-jitter, jitter)
    return ft(w_ft * factor_w), ft(h_ft * factor_h)

# ----------------- PLACEMENT & OVERLAP MANAGEMENT -----------------
def place_rect(x, y, w, h):
    return Rect(x, y, w, h)

def try_resolve_overlap(new: Rect, existing: list, safe_margin=ft(1.5), max_iters=300):
    # Attempt sliding along x and y iteratively to remove overlaps.
    r = Rect(new.x, new.y, new.w, new.h)
    for it in range(max_iters):
        collided = None
        for e in existing:
            re = Rect(e['x'], e['y'], e['w'], e['h'])
            if rect_overlap(r, re):
                collided = re
                break
        if not collided:
            return r
        # push new away — vector from center of collided to center of new
        cx_new = r.x + r.w/2
        cy_new = r.y + r.h/2
        cx_e = collided.x + collided.w/2
        cy_e = collided.y + collided.h/2
        dx = cx_new - cx_e
        dy = cy_new - cy_e
        # prioritize smaller push on smaller overlap axis
        push_x = safe_margin if dx >= 0 else -safe_margin
        push_y = safe_margin if dy >= 0 else -safe_margin
        if abs(dx) > abs(dy):
            r = Rect(r.x + push_x, r.y, r.w, r.h)
        else:
            r = Rect(r.x, r.y + push_y, r.w, r.h)
    return r  # best-effort

# ----------------- SHARED WALLS, DOORS, WINDOWS -----------------
def shared_wall_segment(a, b):
    # returns (orientation, (x0,y0),(x1,y1)) or (None,None,None)
    ax0, ax1 = a['x'], a['x'] + a['w']
    ay0, ay1 = a['y'], a['y'] + a['h']
    bx0, bx1 = b['x'], b['x'] + b['w']
    by0, by1 = b['y'], b['y'] + b['h']
    eps = 1e-3
    # a right wall touches b left wall
    if abs(ax1 - bx0) < eps:
        oy0, oy1 = max(ay0, by0), min(ay1, by1)
        if oy1 - oy0 >= ft(MIN_CLEAR_FT):
            return 'vertical', (ax1, oy0), (ax1, oy1)
    if abs(ax0 - bx1) < eps:
        oy0, oy1 = max(ay0, by0), min(ay1, by1)
        if oy1 - oy0 >= ft(MIN_CLEAR_FT):
            return 'vertical', (ax0, oy0), (ax0, oy1)
    if abs(ay1 - by0) < eps:
        ox0, ox1 = max(ax0, bx0), min(ax1, bx1)
        if ox1 - ox0 >= ft(MIN_CLEAR_FT):
            return 'horizontal', (ox0, ay1), (ox1, ay1)
    if abs(ay0 - by1) < eps:
        ox0, ox1 = max(ax0, bx0), min(ax1, bx1)
        if ox1 - ox0 >= ft(MIN_CLEAR_FT):
            return 'horizontal', (ox0, ay0), (ox1, ay0)
    return None, None, None

def place_doors(rooms):
    doors = []
    for i in range(len(rooms)):
        for j in range(i+1, len(rooms)):
            seg = shared_wall_segment(rooms[i], rooms[j])
            if seg[0] is None: continue
            orient, (x0,y0), (x1,y1) = seg
            length = math.hypot(x1-x0, y1-y0)
            # leave margins at ends for structure
            margin = ft(1.0)
            avail = length - 2*margin
            if avail < door_width():
                continue
            # place door centered in shared segment with small random offset
            center_offset = rnd.uniform(-avail*0.2, avail*0.2)
            if orient == 'vertical':
                mx = x0
                my = (y0 + y1)/2 + center_offset
                side_on = 'vertical'
            else:
                mx = (x0 + x1)/2 + center_offset
                my = y0
                side_on = 'horizontal'
            doors.append({
                'between': (rooms[i]['name'], rooms[j]['name']),
                'pos': [mx, my],
                'width': door_width(),
                'orient': side_on
            })
    return doors

def place_windows(rooms, exterior_buffer=ft(2.0)):
    # naive: any wall portion longer than WINDOW_MIN_FT that is not shared -> window
    windows = []
    for r in rooms:
        # examine four edges and check if exterior by seeing if any other room touches that edge
        edges = [
            ('left', r['x'], r['y'], r['x'], r['y']+r['h']),
            ('right', r['x']+r['w'], r['y'], r['x']+r['w'], r['y']+r['h']),
            ('bottom', r['x'], r['y'], r['x']+r['w'], r['y']),
            ('top', r['x'], r['y']+r['h'], r['x']+r['w'], r['y']+r['h'])
        ]
        for side, x0,y0,x1,y1 in edges:
            is_shared = False
            seg_len = math.hypot(x1-x0, y1-y0)
            if seg_len < ft(WINDOW_MIN_FT): continue
            for other in rooms:
                if other is r: continue
                orient, a0, a1 = shared_wall_segment(r, other)
                if orient:
                    # if share and segment overlaps with this edge -> not exterior
                    is_shared = True
                    break
            if not is_shared:
                # place 1-2 windows along this wall
                count = 1 if seg_len < ft(8) else 2
                for i in range(count):
                    frac = (i+1)/(count+1)
                    wx = x0 + frac*(x1-x0)
                    wy = y0 + frac*(y1-y0)
                    windows.append({'room': r['name'], 'pos': [wx,wy], 'orient': side, 'width': ft(4)})
    return windows

# ----------------- FURNITURE PLACEMENT -----------------
def bbox_overlap(a, b):
    return not (a['x2'] <= b['x1'] or b['x2'] <= a['x1'] or a['y2'] <= b['y1'] or b['y2'] <= a['y1'])

def place_furniture(rooms):
    items = []
    occupied = []
    for r in rooms:
        fdefs = FURNITURE.get(r['type'], [])
        # place along inner perimeter with padding from walls
        pad = ft(1.0)
        for name, fw_ft, fh_ft in fdefs:
            fw, fh = ft(fw_ft), ft(fh_ft)
            attempts = 0
            placed = False
            # try positions: along top, bottom, left, right, then center
            candidates = []
            # top
            candidates.append((r['x'] + r['w']/2 - fw/2, r['y'] + r['h'] - fh - pad))
            # bottom
            candidates.append((r['x'] + r['w']/2 - fw/2, r['y'] + pad))
            # left
            candidates.append((r['x'] + pad, r['y'] + r['h']/2 - fh/2))
            # right
            candidates.append((r['x'] + r['w'] - fw - pad, r['y'] + r['h']/2 - fh/2))
            # center
            candidates.append((r['x'] + r['w']/2 - fw/2 + rnd.uniform(-ft(0.5), ft(0.5)),
                               r['y'] + r['h']/2 - fh/2 + rnd.uniform(-ft(0.5), ft(0.5))))
            for cx, cy in candidates:
                attempts += 1
                box = {'x1': cx, 'y1': cy, 'x2': cx + fw, 'y2': cy + fh}
                collision = False
                # keep inside room
                if box['x1'] < r['x'] + pad or box['y1'] < r['y'] + pad or box['x2'] > r['x'] + r['w'] - pad or box['y2'] > r['y'] + r['h'] - pad:
                    collision = True
                else:
                    for oc in occupied:
                        if bbox_overlap(box, oc):
                            collision = True
                            break
                if not collision:
                    occupied.append(box)
                    items.append({'type': name, 'room': r['name'], 'pos': [box['x1'] + fw/2, box['y1'] + fh/2], 'size':[fw,fh]})
                    placed = True
                    break
            if not placed:
                # place at random safe spot inside room
                for _ in range(10):
                    cx = rnd.uniform(r['x'] + pad, r['x'] + r['w'] - fw - pad)
                    cy = rnd.uniform(r['y'] + pad, r['y'] + r['h'] - fh - pad)
                    box = {'x1': cx, 'y1': cy, 'x2': cx + fw, 'y2': cy + fh}
                    collision = any(bbox_overlap(box, oc) for oc in occupied)
                    if not collision:
                        occupied.append(box)
                        items.append({'type': name, 'room': r['name'], 'pos': [box['x1'] + fw/2, box['y1'] + fh/2], 'size':[fw,fh]})
                        break
    return items

# ----------------- PLAN GENERATION -----------------
def generate_plan(config, meta=None):
    meta = meta or {}
    rooms = []

    # seed option
    if 'seed' in meta:
        rnd.seed(meta['seed'])

    # place living room at origin-ish center
    cx, cy = 400, 300
    lw, lh = sample_room_dims('Living Room')
    living = place_rect(cx - lw/2, cy - lh/2, lw, lh)
    rooms.append(rect_to_dict(living, 'Living Room', 'Living Room'))

    # dining attached to right of living
    if config.get('Dining Room', 0) > 0:
        dw, dh = sample_room_dims('Dining Room')
        dining = place_rect(living.x + living.w + ft(0.5), living.y + (living.h - dh)/2, dw, dh)
        dining = try_resolve_overlap(dining, rooms)
        rooms.append(rect_to_dict(dining, 'Dining Room', 'Dining Room'))

    # kitchen attached to dining or living (right of dining preferred)
    if config.get('Kitchen', 0) > 0:
        ref = None
        for r in rooms:
            if r['type']=='Dining Room':
                ref = r
                break
        if not ref:
            ref = rooms[0]  # living
        kw, kh = sample_room_dims('Kitchen')
        kitchen = place_rect(ref['x'] + ref['w'] + ft(0.5), ref['y'] + (ref['h'] - kh)/2, kw, kh)
        kitchen = try_resolve_overlap(kitchen, rooms)
        rooms.append(rect_to_dict(kitchen, 'Kitchen', 'Kitchen'))

    # hallway below living to serve bedrooms
    hall_w = living.w if isinstance(living, Rect) else ft(ROOM_NOMINAL['Living Room'][0])
    hall_h = ft(ROOM_NOMINAL['Hall'][1])
    hall = place_rect(living.x, living.y - hall_h - ft(1.0), hall_w, hall_h)
    hall = try_resolve_overlap(hall, rooms)
    rooms.append(rect_to_dict(hall, 'Hall', 'Hall'))

    # bedrooms line along bottom of hall
    bcount = config.get('Bedroom', 0)
    bed_w, bed_h = sample_room_dims('Bedroom')
    # place bedrooms left-to-right under hall with small gaps
    start_x = hall.x
    gap = ft(1.0)
    curx = start_x
    for i in range(bcount):
        bx = curx
        by = hall.y - bed_h - ft(0.5)
        room = place_rect(bx, by, bed_w, bed_h)
        room = try_resolve_overlap(room, rooms)
        rooms.append(rect_to_dict(room, 'Bedroom', f'Bedroom {i+1}'))
        curx = room.x + room.w + gap

    # bathrooms: attach to bedrooms (top side) or nearby if none
    bneeded = config.get('Bathroom', 0)
    b_placed = 0
    for i in range(bneeded):
        target_beds = [r for r in rooms if r['type']=='Bedroom']
        if not target_beds:
            break
        tgt = target_beds[min(i, len(target_beds)-1)]
        bw, bh = sample_room_dims('Bathroom')
        # attach to top of bedroom (between bedroom and hall)
        bath = place_rect(tgt['x'] + (tgt['w'] - bw)/2, tgt['y'] + tgt['h'] + ft(0.1), bw, bh)
        bath = try_resolve_overlap(bath, rooms)
        rooms.append(rect_to_dict(bath, 'Bathroom', f'Bathroom {i+1}'))
        b_placed += 1

    # laundry attach near kitchen or tucked near bath
    if config.get('Laundry', 0) > 0:
        ref = next((r for r in rooms if r['type']=='Kitchen'), None) or next((r for r in rooms if r['type']=='Bathroom'), None) or rooms[0]
        lw_w, lw_h = sample_room_dims('Laundry')
        laund = place_rect(ref['x'] + ref['w'] + ft(0.5), ref['y'], lw_w, lw_h)
        laund = try_resolve_overlap(laund, rooms)
        rooms.append(rect_to_dict(laund, 'Laundry', 'Laundry'))

    # post-process: avoid tiny overlaps and enforce minimal separation for non-adjacent types
    cleaned = []
    for r in rooms:
        rr = Rect(r['x'], r['y'], r['w'], r['h'])
        rr = try_resolve_overlap(rr, cleaned)
        cleaned.append(rect_to_dict(rr, r['type'], r['name']))

    rooms = cleaned

    # Doors/windows/furniture
    doors = place_doors(rooms)
    windows = place_windows(rooms)
    furniture = place_furniture(rooms)

    plan = {
        'rooms': rooms,
        'doors': doors,
        'windows': windows,
        'furniture': furniture,
        'meta': meta
    }
    return plan

# ----------------- DRAW / RENDER -----------------
def draw_plan(plan, fname='procedural_floorplan.png', show=True):
    rooms = plan['rooms']
    doors = plan.get('doors', [])
    windows = plan.get('windows', [])
    furniture = plan.get('furniture', [])

    minx = min(r['x'] for r in rooms) - ft(4)
    miny = min(r['y'] for r in rooms) - ft(4)
    maxx = max(r['x'] + r['w'] for r in rooms) + ft(4)
    maxy = max(r['y'] + r['h'] for r in rooms) + ft(4)

    fig_w = max(6, (maxx-minx)/120)
    fig_h = max(6, (maxy-miny)/120)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect('equal')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis('off')
    ax.set_facecolor('#f7f7f7')

    # Draw rooms (floor fill)
    for r in rooms:
        ax.add_patch(patches.Rectangle((r['x'], r['y']), r['w'], r['h'], facecolor='#f0fff4', edgecolor='black', linewidth=0.8))
        ax.text(r['x'] + 6, r['y'] + r['h'] - 12, r['name'], ha='left', va='top', fontsize=9, weight='bold')

    # Draw walls (inner border thicker)
    for r in rooms:
        # outer wall rectangle with thickness shown
        wt = wall_thickness()
        # draw inner room rectangle already drawn; show wall edges as thicker lines
        ax.add_patch(patches.Rectangle((r['x'], r['y']), r['w'], r['h'], fill=False, linewidth=2, edgecolor='#333'))

    # Draw doors
    for d in doors:
        x, y = d['pos']
        w = d['width']
        if d['orient'] == 'vertical':
            ax.plot([x, x], [y - w/2, y + w/2], color='#8b5a2b', linewidth=6, solid_capstyle='butt')
        else:
            ax.plot([x - w/2, x + w/2], [y, y], color='#8b5a2b', linewidth=6, solid_capstyle='butt')

    # Draw windows
    for w in windows:
        x,y = w['pos']
        if w['orient'] in ('left','right'):
            ax.plot([x, x], [y - ft(1), y + ft(1)], color='#5dade2', linewidth=3)
        else:
            ax.plot([x - ft(1), x + ft(1)], [y, y], color='#5dade2', linewidth=3)

    # Draw furniture
    for f in furniture:
        fx, fy = f['pos']
        fw, fh = f['size']
        color = FURN_COLORS.get(f['type'], 'gray')
        ax.add_patch(patches.Rectangle((fx - fw/2, fy - fh/2), fw, fh, facecolor=color, edgecolor='black', alpha=0.95))
        ax.text(fx, fy, f['type'], ha='center', va='center', fontsize=6, color='white')

    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    if show:
        plt.show()

# ----------------- RUN BLOCK (with ArchitectAgent support) -----------------
if __name__ == '__main__':
    import argparse
    import random as _rnd

    parser = argparse.ArgumentParser(description='Procedural floor plan generator (AAI improved)')
    parser.add_argument('--auto', action='store_true', help='Use default config')
    parser.add_argument('--agent', type=str, help='Use ArchitectAgent with goal text (e.g. "3 bed 2 bath")')
    parser.add_argument('--out', type=str, default='procedural_floorplan.png', help='Output image filename')
    parser.add_argument('--no-show', action='store_true', help='Do not show interactive plot window')
    args = parser.parse_args()

    if args.auto:
        config = {'Living Room': 1, 'Dining Room': 1, 'Kitchen': 1, 'Bedroom': 3, 'Bathroom': 2, 'Laundry': 1}
        meta = {}
    elif args.agent:
        try:
            from architect_agent import ArchitectAgent
            agent = ArchitectAgent()
            res = agent.interpret(args.agent)
            config = res.get('config', {})
            meta = res.get('meta', {})
            if 'seed' in meta:
                _rnd.seed(meta['seed'])
            print("Agent config:", config, "meta:", meta)
        except Exception as e:
            print("ArchitectAgent not available or error:", e)
            config = {'Living Room': 1, 'Dining Room': 1, 'Kitchen': 1, 'Bedroom': 2, 'Bathroom': 1, 'Laundry': 0}
            meta = {}
    else:
        print("\nConfigure rooms (press enter for 0):")
        keys = ['Living Room', 'Dining Room', 'Kitchen', 'Bedroom', 'Bathroom', 'Laundry']
        config = {}
        for k in keys:
            while True:
                try:
                    val = input(f"{k} count (default 0): ").strip()
                    if val == '':
                        config[k] = 0
                        break
                    config[k] = max(0, int(val))
                    break
                except ValueError:
                    print("Enter integer.")
        meta = {}

    plan = generate_plan(config, meta=meta)
    with open('procedural_floorplan.json','w') as f:
        json.dump(plan, f, indent=2)
    draw_plan(plan, fname=args.out, show=not args.no_show)
