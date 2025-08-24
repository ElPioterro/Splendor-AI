import pygame
from typing import Optional, List

from game_engine import (
    GameState, Move, GemColor, Card, Noble, Player,
    TakeThreeGems, TakeTwoGems, ReserveVisibleCard, ReserveFromDeck, BuyCard
)

SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 900
HEADER_HEIGHT = 60
PLAYER_PANEL_HEIGHT = 150
TOKENS_AREA_WIDTH = 280
MARGIN = 20

BACKGROUND_COLOR, PANEL_COLOR = (20, 40, 60), (40, 60, 80)
FONT_COLOR = (220, 220, 220)
ACTIVE_PLAYER_BORDER_COLOR = (255, 215, 0)

GEM_COLORS_RGB = {
    GemColor.BLUE: (0, 100, 255), GemColor.GREEN: (0, 150, 50),
    GemColor.RED: (200, 0, 0), GemColor.WHITE: (230, 230, 230),
    GemColor.BLACK: (30, 30, 30), "gold": (255, 215, 0)
}

CARD_WIDTH, CARD_HEIGHT = 150, 210
NOBLE_WIDTH, NOBLE_HEIGHT = 120, 120

COLOR_ORDER = [GemColor.WHITE, GemColor.BLUE, GemColor.GREEN, GemColor.RED, GemColor.BLACK]


class SplendorGUI:
    def __init__(self) -> None:
        pygame.init(); pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Splendor AI - PROMETHEUS")
        self.f_small = pygame.font.SysFont("Arial", 16)
        self.f_medium = pygame.font.SysFont("Arial", 22)
        self.f_large = pygame.font.SysFont("Arial", 30, bold=True)
        self.layout = {}                      # metryka aktualnej klatki
        self.reserved_hitboxes = {}           # {player_index: [(rect, card), ...]}
        self.bottom_panel_rects = {}          # {player_index: rect}

        # Animations:
        self.card_slot_flashes = []  # list[dict]: {'tier':int, 'index':int, 'start':ms, 'duration':ms}
        self.slot_rects = {1: {}, 2: {}, 3: {}}  # mapowanie (tier->index->rect) aktualnej klatki
        self.token_deltas = []  # list[dict]: {'key': GemColor|'gold', 'value': int, 'start':ms, 'duration':ms}
        self.token_centers = {}  # mapowanie koloru licznika -> (x, y)
        
        self.fx_flash_ms = 1100       # czas flasha slotu karty (ms)
        self.fx_delta_ms = 1600       # czas pływających delt przy licznikach (ms)
        self.fx_delta_drift = 14      # pionowy dryf (px) w czasie trwania efektu
                
        print("GUI v2.0 ready")

        # HITBOXY/stan rysowania
        self.token_hitboxes: List[tuple[pygame.Rect, GemColor]] = []
        self.visible_card_hitboxes = {1: [], 2: [], 3: []}  # list[(rect, card)]
        self.deck_hitboxes = {}  # tier -> rect
        self.noble_choice_hitboxes: List[tuple[pygame.Rect, Noble]] = []
        self.noble_cancel_rect: Optional[pygame.Rect] = None
        self.right_panel_rect: Optional[pygame.Rect] = None

    def _ease_out(self, t: float) -> float:
        # 0..1 -> 0..1, wolniej na końcu (cubic)
        t = max(0.0, min(1.0, t))
        return 1 - (1 - t) ** 3

    def _compute_layout(self):
        W, H = self.screen.get_size()
        start_x = MARGIN
        start_y = HEADER_HEIGHT + MARGIN
        bottom_top = H - PLAYER_PANEL_HEIGHT - MARGIN  # górna krawędź paneli graczy
        available_h = bottom_top - start_y
        base_needed = NOBLE_HEIGHT + MARGIN + 3 * CARD_HEIGHT + 2 * MARGIN
        s = min(1.0, available_h / base_needed)  # skala pionowa, max 1.0

        gap = max(10, int(MARGIN * s))
        noble_w, noble_h = int(NOBLE_WIDTH * s), int(NOBLE_HEIGHT * s)
        card_w, card_h = int(CARD_WIDTH * s), int(CARD_HEIGHT * s)
        card_start_y = start_y + noble_h + gap
        deck_x = start_x + 4 * (card_w + gap)

        tokens_area_x = SCREEN_WIDTH - TOKENS_AREA_WIDTH - MARGIN
        inspector_x = deck_x + card_w + gap
        inspector_w = max(0, tokens_area_x - inspector_x - MARGIN)
        inspector_rect = pygame.Rect(inspector_x, card_start_y, inspector_w, bottom_top - card_start_y)

        self.layout = dict(
            s=s, start_x=start_x, start_y=start_y, bottom_top=bottom_top,
            gap=gap, noble_w=noble_w, noble_h=noble_h,
            card_w=card_w, card_h=card_h,
            card_start_y=card_start_y, deck_x=deck_x,
            inspector_rect=inspector_rect
        )

    def _draw_header(self, gs: GameState, status_message: str):
        current_player = gs.get_current_player()
        pygame.draw.rect(self.screen, PANEL_COLOR, (0, 0, SCREEN_WIDTH, HEADER_HEIGHT))
        txt_turn = self.f_medium.render(f"Tura: {current_player.name}", True, FONT_COLOR)
        txt_title = self.f_large.render("Splendor AI", True, FONT_COLOR)
        self.screen.blit(txt_turn, (MARGIN, 15))
        self.screen.blit(txt_title, (SCREEN_WIDTH // 2 - txt_title.get_width() // 2, 10))
        if status_message:
            msg_surface = self.f_medium.render(status_message, True, (255, 100, 100))
            self.screen.blit(msg_surface, (SCREEN_WIDTH - msg_surface.get_width() - MARGIN, 15))

    def _draw_single_card(self, obj, rect: pygame.Rect, is_noble=False):
        card_color = (180, 140, 80) if is_noble else (60, 80, 100)
        pygame.draw.rect(self.screen, card_color, rect, border_radius=5)
        
        title = f"#{getattr(obj, 'id', '')}" if hasattr(obj, 'id') else ""
        self.screen.blit(self.f_small.render(title, True, FONT_COLOR), (rect.x + 5, rect.y + 5))

        if obj.prestige_points > 0:
            pts_txt = self.f_medium.render(str(obj.prestige_points), True, (10, 10, 10))
            pygame.draw.circle(self.screen, (255, 215, 0), (rect.right - 20, rect.top + 20), 12)
            self.screen.blit(pts_txt, (rect.right - 20 - pts_txt.get_width()//2, rect.top + 20 - pts_txt.get_height()//2))

        if isinstance(obj, Card):
            pygame.draw.rect(self.screen, GEM_COLORS_RGB[obj.bonus_color], (rect.x + 5, rect.y + 35, rect.w - 10, 8), border_radius=3)
        
        # --- robust: cost/requirements mogą być dict lub listami krotek ---
        cost_y = rect.bottom - 25
        raw_costs = obj.cost if isinstance(obj, Card) else obj.requirements
        if isinstance(raw_costs, dict):
            items = list(raw_costs.items())
        else:
            items = []
            for entry in (raw_costs or []):
                if isinstance(entry, (list, tuple)):
                    if len(entry) >= 2:
                        items.append((entry[0], entry[1]))
                elif isinstance(entry, dict):
                    col = entry.get('color') or entry.get('gem') or entry.get('name')
                    amt = entry.get('amount') or entry.get('count') or 0
                    if col is None:
                        continue
                    items.append((col, amt))

        for i, (color, amount) in enumerate(items):
            if not amount:
                continue
            if isinstance(color, str):
                try:
                    color = GemColor[color.upper()]
                except Exception:
                    continue
            cost_x = rect.x + 20 + i * 28
            pygame.draw.circle(self.screen, GEM_COLORS_RGB[color], (cost_x, cost_y), 10)
            txt = self.f_small.render(str(amount), True, (0,0,0) if color == GemColor.WHITE else FONT_COLOR)
            self.screen.blit(txt, (cost_x - txt.get_width()//2, cost_y - txt.get_height()//2))

    def _draw_card_slot_flashes(self):
        now = pygame.time.get_ticks()
        alive = []
        for fx in self.card_slot_flashes:
            t, idx, start, dur = fx['tier'], fx['index'], fx['start'], fx['duration']
            rect = self.slot_rects.get(t, {}).get(idx)
            if not rect:
                if now - start < dur:
                    alive.append(fx)
                continue
            raw = (now - start) / max(1, dur)
            p = self._ease_out(raw)
            if raw >= 1.0:
                continue
            alpha = int(190 * (1.0 - p))  # było ok. 160 → wolniejsze, dłużej widoczne
            overlay = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            overlay.fill((255, 255, 0, alpha))
            self.screen.blit(overlay, (rect.x, rect.y))
            pygame.draw.rect(self.screen, (255, 215, 0), rect, width=2, border_radius=6)
            alive.append(fx)
        self.card_slot_flashes = alive
    
    def _draw_cards_area(self, gs: GameState):
        L = self.layout
        start_x, start_y = L['start_x'], L['start_y']
        gap = L['gap']
        noble_w, noble_h = L['noble_w'], L['noble_h']
        card_w, card_h = L['card_w'], L['card_h']
        card_start_y = L['card_start_y']
        deck_x = L['deck_x']


        # wyczyść hitboxy
        self.visible_card_hitboxes = {1: [], 2: [], 3: []}
        self.deck_hitboxes = {}
        self.slot_rects = {1: {}, 2: {}, 3: {}}

        # Arystokraci
        for i, noble in enumerate(gs.nobles):
            rect = pygame.Rect(start_x + i * (noble_w + gap), start_y, noble_w, noble_h)
            self._draw_single_card(noble, rect, is_noble=True)

        # Widoczne karty + stos
        for tier in (3, 2, 1):
            y_pos = card_start_y + (3 - tier) * (card_h + gap)

            deck_rect = pygame.Rect(deck_x, y_pos, card_w, card_h)
            pygame.draw.rect(self.screen, (80, 80, 80), deck_rect, border_radius=5)
            lab = self.f_medium.render(f"T{tier}", True, FONT_COLOR)
            self.screen.blit(lab, (deck_rect.centerx - lab.get_width()//2,
                                deck_rect.centery - lab.get_height()//2))
            self.deck_hitboxes[tier] = deck_rect

            for col, card in enumerate(gs.visible_cards.get(tier, [])):
                rect = pygame.Rect(start_x + col * (card_w + gap), y_pos, card_w, card_h)
                self._draw_single_card(card, rect)
                self.visible_card_hitboxes[tier].append((rect, card))
                self.slot_rects[tier][col] = rect  # <-- zapamiętaj slot
        
        self._draw_card_slot_flashes()

    def _draw_token_delta_fx(self):
        now = pygame.time.get_ticks()
        alive = []
        for fx in self.token_deltas:
            key, val, start, dur = fx['key'], fx['value'], fx['start'], fx['duration']
            center = self.token_centers.get(key)
            if not center:
                if now - start < dur:
                    alive.append(fx)
                continue
            raw = (now - start) / max(1, dur)
            p = self._ease_out(raw)
            if raw >= 1.0:
                continue
            base_col = (255, 80, 80) if val < 0 else (80, 220, 120)
            alpha = int(255 * (1.0 - p))        # dłuższe wygaszanie
            dy = int(-self.fx_delta_drift * p)  # wolniejszy ruch w górę
            txt = f"{val:+d}"
            surf = self.f_medium.render(txt, True, base_col).convert_alpha()
            surf.set_alpha(alpha)
            self.screen.blit(surf, (center[0] + 32, center[1] - 10 + dy))
            alive.append(fx)
        self.token_deltas = alive

    def _draw_tokens_area(self, gs: GameState):
        area_x = SCREEN_WIDTH - TOKENS_AREA_WIDTH - MARGIN
        area_rect = pygame.Rect(area_x, HEADER_HEIGHT + MARGIN, TOKENS_AREA_WIDTH, SCREEN_HEIGHT - HEADER_HEIGHT - PLAYER_PANEL_HEIGHT - 3*MARGIN)
        pygame.draw.rect(self.screen, PANEL_COLOR, area_rect)
        self.right_panel_rect = area_rect
        self.token_centers = {}

        # hitboxy żetonów
        self.token_hitboxes = []
        y = area_rect.y + MARGIN + 10
        for color in GemColor:
            if gs.available_gems[color] > 0:
                center = (area_x + 50, y)
                pygame.draw.circle(self.screen, GEM_COLORS_RGB[color], center, 25)
                amount = self.f_large.render(f"x{gs.available_gems[color]}", True, FONT_COLOR)
                self.screen.blit(amount, (area_x + 100, y - 15))
                self.token_hitboxes.append((pygame.Rect(center[0]-25, center[1]-25, 50, 50), color))
                self.token_centers[color] = center
                y += 60

        if gs.gold_gems > 0:
            center = (area_x + 50, y)
            pygame.draw.circle(self.screen, GEM_COLORS_RGB["gold"], center, 25)
            self.screen.blit(self.f_large.render(f"x{gs.gold_gems}", True, FONT_COLOR), (area_x + 100, y - 15))
            self.token_centers['gold'] = center
            # gold nie jest GemColor – celowo nie dodajemy do hitboxów
            y += 60

        self._draw_token_delta_fx()
        # zwracamy y kończące sekcję żetonów (przyda się dla paneli poniżej)
        return y + MARGIN

    def push_token_deltas(self, deltas: dict, duration: int | None = None):
        if duration is None:
            duration = self.fx_delta_ms
        t = pygame.time.get_ticks()
        for key, val in deltas.items():
            if not val:
                continue
            self.token_deltas.append({'key': key, 'value': int(val), 'start': t, 'duration': duration})

    def _draw_ai_panel_right(self, opponent: Player, start_y: int):
        if not self.right_panel_rect:
            return
        x = self.right_panel_rect.x + 10
        y = max(start_y, self.right_panel_rect.y + 10)
        # Prosty panel AI
        title = self.f_medium.render(f"Przeciwnik: {opponent.name}", True, FONT_COLOR)
        self.screen.blit(title, (x, y)); y += 24
        pts = self.f_small.render(f"Punkty: {opponent.prestige_points}", True, FONT_COLOR)
        self.screen.blit(pts, (x, y)); y += 18
        gtxt = "Żetony: " + " ".join(f"{c.name[0]}:{n}" for c,n in opponent.gems.items() if n) + (f" Zł:{opponent.gold_gems}" if opponent.gold_gems else "")
        self.screen.blit(self.f_small.render(gtxt, True, FONT_COLOR), (x, y)); y += 18
        btxt = "Bonusy: " + " ".join(f"{c.name[0]}:{n}" for c,n in opponent.bonuses.items() if n)
        self.screen.blit(self.f_small.render(btxt, True, FONT_COLOR), (x, y)); y += 18
        rtxt = f"Zarezerw.: {len(opponent.reserved_cards)}"
        self.screen.blit(self.f_small.render(rtxt, True, FONT_COLOR), (x, y))

    def _draw_log_panel(self, log_lines: List[str], start_y: int):
        if not self.right_panel_rect or not log_lines:
            return
        # panel logów na dole prawego panelu
        x = self.right_panel_rect.x + 10
        bottom = self.right_panel_rect.bottom - 10
        # wyświetlamy od dołu do góry ostatnie 6-9 wpisów
        max_lines = 9
        lines = log_lines[-max_lines:]
        for i, line in enumerate(reversed(lines)):
            y = bottom - i * 18
            surf = self.f_small.render(line, True, (200, 220, 255))
            self.screen.blit(surf, (x, y - surf.get_height()))

    def _draw_single_player_panel(self, player: Player, rect: pygame.Rect, is_active: bool,
                                    selected_gems: List[GemColor], player_index: int):
        pygame.draw.rect(self.screen, PANEL_COLOR, rect)
        if is_active:
            pygame.draw.rect(self.screen, ACTIVE_PLAYER_BORDER_COLOR, rect, 4, border_radius=6)

        # header
        x, y = rect.x + 12, rect.y + 8
        self.screen.blit(self.f_medium.render(player.name, True, FONT_COLOR), (x, y))
        pts = self.f_large.render(f"{player.prestige_points} Pkt.", True, FONT_COLOR)
        self.screen.blit(pts, (rect.right - pts.get_width() - 12, y))

        # rząd bonusów ("Karty")
        y += 32
        self.screen.blit(self.f_small.render("Karty:", True, FONT_COLOR), (x, y))
        bx = x + 60
        for color in COLOR_ORDER:
            box = pygame.Rect(bx, y - 2, 18, 18)
            pygame.draw.rect(self.screen, GEM_COLORS_RGB[color], box, border_radius=3)
            val = str(player.bonuses.get(color, 0))
            self.screen.blit(self.f_small.render(val, True, FONT_COLOR), (bx + 20, y - 2))
            bx += 50

        # rząd żetonów
        y += 24

        total = player.total_gems
        if total >= 10:
            limit_col = (255, 80, 80)
        elif total >= 8:
            limit_col = (255, 180, 0)
        else:
            limit_col = (200, 220, 255)

        label = self.f_small.render("Żetony:", True, FONT_COLOR)
        self.screen.blit(label, (x, y))

        limit_badge = self.f_small.render(f"({total}/10)", True, limit_col)
        self.screen.blit(limit_badge, (x + 70, y))

        tx = x + 120  # przesuwamy start rzędu żetonów w prawo, żeby zrobić miejsce na badge

        for color in COLOR_ORDER:
            pygame.draw.circle(self.screen, GEM_COLORS_RGB[color], (tx, y + 7), 9)
            val = str(player.gems.get(color, 0))
            self.screen.blit(self.f_small.render(val, True, FONT_COLOR), (tx + 14, y - 2))
            tx += 48
        # złoto
        pygame.draw.circle(self.screen, GEM_COLORS_RGB["gold"], (tx, y + 7), 9)
        self.screen.blit(self.f_small.render(str(player.gold_gems), True, FONT_COLOR), (tx + 14, y - 2))

        # rząd rezerw
        y += 26
        self.screen.blit(self.f_small.render("Reserve:", True, FONT_COLOR), (x, y))
        rx = x + 70
        rw, rh = int(CARD_WIDTH * 0.36), int(CARD_HEIGHT * 0.36)
        # zapisz hitboxy rezerw tego gracza
        self.reserved_hitboxes[player_index] = []
        for i, card in enumerate(player.reserved_cards):
            r = pygame.Rect(rx + i * (rw + 8), y - 8, rw, rh)
            self._draw_single_card(card, r)
            self.reserved_hitboxes[player_index].append((r, card))

        # (opcjonalnie) małe kropki z aktualnym wyborem żetonów dla aktywnego gracza
        if is_active and selected_gems:
            sx = rect.right - 140
            sy = rect.bottom - 20
            self.screen.blit(self.f_small.render("Wybór:", True, FONT_COLOR), (sx - 55, sy - 9))
            for i, color in enumerate(selected_gems):
                pygame.draw.circle(self.screen, GEM_COLORS_RGB[color], (sx + i * 20, sy), 7)

    def _extract_cost_dict(self, raw_costs):
        # ujednolicenie do dict[GemColor, int]
        if isinstance(raw_costs, dict):
            return dict(raw_costs)
        d = {c: 0 for c in GemColor}
        for entry in (raw_costs or []):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                col, amt = entry[0], entry[1]
            elif isinstance(entry, dict):
                col = entry.get('color') or entry.get('gem') or entry.get('name')
                amt = entry.get('amount') or entry.get('count') or 0
            else:
                continue
            if isinstance(col, str):
                try: col = GemColor[col.upper()]
                except Exception: continue
            d[col] = d.get(col, 0) + int(amt or 0)
        return d

    def _pick_hovered_card(self, mx, my):
        # karty na stole
        for tier in (3, 2, 1):
            for rect, card in self.visible_card_hitboxes[tier]:
                if rect.collidepoint(mx, my):
                    return card
        # rezerwy (oba panele)
        for plist in self.reserved_hitboxes.values():
            for rect, card in plist:
                if rect.collidepoint(mx, my):
                    return card
        return None

    def flash_card_slot(self, tier: int, index: int, duration: int | None = None):
        if duration is None:
            duration = self.fx_flash_ms
        self.card_slot_flashes.append({
            'tier': tier, 'index': index,
            'start': pygame.time.get_ticks(),
            'duration': duration
        })

    def _draw_inspector(self, gs: GameState, human_player_index: int):
        L = self.layout
        rect = L.get('inspector_rect')
        if not rect or rect.w < 120 or rect.h < 140:
            return

        # tło panelu
        pygame.draw.rect(self.screen, PANEL_COLOR, rect, border_radius=6)
        title = self.f_medium.render("Inspektor", True, FONT_COLOR)
        self.screen.blit(title, (rect.x + 10, rect.y + 8))

        mx, my = pygame.mouse.get_pos()
        card = self._pick_hovered_card(mx, my)
        if not card:
            # wskazówki
            tips = [
                "Najedź na kartę, aby zobaczyć podgląd i koszt.",
                "LPM na karcie: kup",
                "PPM na karcie: rezerwuj",
                "PPM na stosie T1/T2/T3: rezerwuj ze stosu"
            ]
            y = rect.y + 36
            for t in tips:
                surf = self.f_small.render(t, True, (200, 220, 255))
                self.screen.blit(surf, (rect.x + 10, y))
                y += 20
            return

        # podgląd karty
        gap = L['gap']
        # dopasuj podgląd do górnej połowy panelu
        max_w = rect.w - 2 * gap
        max_h = int(rect.h * 0.55) - gap
        pw = min(max_w, int(L['card_w'] * 1.15))
        ph = int(pw * CARD_HEIGHT / CARD_WIDTH)
        if ph > max_h:
            ph = max_h
            pw = int(ph * CARD_WIDTH / CARD_HEIGHT)
        preview_rect = pygame.Rect(rect.x + (rect.w - pw)//2, rect.y + 34, pw, ph)
        self._draw_single_card(card, preview_rect)

        # analiza kosztu vs zasobów gracza
        pl = gs.players[human_player_index]
        costs = self._extract_cost_dict(card.cost)
        y = preview_rect.bottom + gap
        self.screen.blit(self.f_small.render("Koszt vs zasoby gracza:", True, FONT_COLOR),
                        (rect.x + 10, y)); y += 18

        total_deficit = 0
        for color in COLOR_ORDER:
            c = int(costs.get(color, 0))
            bonus = int(pl.bonuses.get(color, 0))
            tokens = int(pl.gems.get(color, 0))
            net = max(0, c - bonus)
            deficit = max(0, net - tokens)
            total_deficit += deficit

            # kropka koloru
            pygame.draw.circle(self.screen, GEM_COLORS_RGB[color], (rect.x + 18, y + 8), 8)
            line = f"koszt:{c}  po bonusie:{net}  masz:{tokens}  brakuje:{deficit}"
            surf = self.f_small.render(line, True, FONT_COLOR)
            self.screen.blit(surf, (rect.x + 32, y))
            y += 18

        # złoto informacyjnie
        gold_info = f"Złoto: {pl.gold_gems}  (może zastąpić brakujące kolory)"
        surf = self.f_small.render(gold_info, True, (255, 215, 0))
        self.screen.blit(surf, (rect.x + 10, y))

    # NOWA FUNKCJA ORGANIZUJĄCA DOLNĄ CZĘŚĆ EKRANU
    def _draw_bottom_panels(self, gs: GameState, human_player_index: int, selected_gems: List[GemColor]):
        panel_width = (SCREEN_WIDTH - 3 * MARGIN) // 2
        panel_y = SCREEN_HEIGHT - PLAYER_PANEL_HEIGHT - MARGIN
        human_rect = pygame.Rect(MARGIN, panel_y, panel_width, PLAYER_PANEL_HEIGHT)
        ai_rect = pygame.Rect(MARGIN * 2 + panel_width, panel_y, panel_width, PLAYER_PANEL_HEIGHT)

        self.bottom_panel_rects[human_player_index] = human_rect
        ai_index = 1 if human_player_index == 0 else 0
        self.bottom_panel_rects[ai_index] = ai_rect

        human_player = gs.players[human_player_index]
        ai_player = None
        for i, p in enumerate(gs.players):
            if i != human_player_index:
                ai_player = p; ai_idx = i; break

        is_human_active = (gs.current_player_index == human_player_index)
        self._draw_single_player_panel(human_player, human_rect, is_active=is_human_active,
                                    selected_gems=selected_gems if is_human_active else [],
                                    player_index=human_player_index)
        if ai_player:
            self._draw_single_player_panel(ai_player, ai_rect, is_active=(not is_human_active),
                                        selected_gems=[], player_index=ai_idx)
    
    def _draw_noble_choice_overlay(self, nobles: List[Noble]):
        # półprzezroczysty cień
        shade = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 160))
        self.screen.blit(shade, (0, 0))

        # panel centralny
        panel_w = max(420, len(nobles) * (NOBLE_WIDTH + MARGIN) + 2*MARGIN)
        panel_h = NOBLE_HEIGHT + 2*MARGIN + 50
        panel_rect = pygame.Rect((SCREEN_WIDTH - panel_w)//2, (SCREEN_HEIGHT - panel_h)//2, panel_w, panel_h)
        pygame.draw.rect(self.screen, (50, 70, 95), panel_rect, border_radius=8)
        title = self.f_medium.render("Wybierz arystokratę", True, FONT_COLOR)
        self.screen.blit(title, (panel_rect.centerx - title.get_width()//2, panel_rect.y + 10))

        # reset hitboxów
        self.noble_choice_hitboxes = []
        # rysuj arystokratów
        base_x = panel_rect.x + MARGIN
        y = panel_rect.y + 40
        for i, noble in enumerate(nobles):
            rect = pygame.Rect(base_x + i * (NOBLE_WIDTH + MARGIN), y, NOBLE_WIDTH, NOBLE_HEIGHT)
            self._draw_single_card(noble, rect, is_noble=True)
            self.noble_choice_hitboxes.append((rect, noble))

        # opcjonalny przycisk "Pomiń"
        btn_rect = pygame.Rect(panel_rect.right - 110, panel_rect.y + 10, 100, 28)
        pygame.draw.rect(self.screen, (90, 110, 130), btn_rect, border_radius=5)
        self.screen.blit(self.f_small.render("Pomiń", True, FONT_COLOR), (btn_rect.centerx - 18, btn_rect.y + 6))
        self.noble_cancel_rect = btn_rect

    def draw_game_state(self, gs: GameState, human_player_index: int, selected_gems: List[GemColor],
                        status_message: str = "", mode: str = "PLAY", eligible_nobles: Optional[List[Noble]] = None,
                        log_lines: Optional[List[str]] = None):
        self.screen.fill(BACKGROUND_COLOR)
        self._compute_layout()
        self._draw_header(gs, status_message)
        self._draw_cards_area(gs)             # używa self.layout (skala/pozycje)
        self._draw_inspector(gs, human_player_index)  # WYPEŁNIA „DZIURĘ”
        tokens_end_y = self._draw_tokens_area(gs)     # prawy panel żetonów
        if log_lines:
            self._draw_log_panel(log_lines, tokens_end_y + 10)

        self._draw_bottom_panels(gs, human_player_index, selected_gems)

        if mode == "CHOOSE_NOBLE" and eligible_nobles:
            self._draw_noble_choice_overlay(eligible_nobles)

        pygame.display.flip()

    def handle_events(self, gs: GameState, mode: str = "PLAY", human_player_index: int = 0) -> Optional[tuple]:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return ("QUIT",)

            if ev.type == pygame.MOUSEBUTTONDOWN:
                x, y = ev.pos

                if mode == "CHOOSE_NOBLE":
                    if ev.button == 1:
                        for rect, noble in self.noble_choice_hitboxes:
                            if rect.collidepoint(x, y):
                                return ("choose_noble", noble)
                        if self.noble_cancel_rect and self.noble_cancel_rect.collidepoint(x, y):
                            return ("choose_noble", None)
                    continue

                if ev.button == 1:
                    # kup widoczną kartę
                    for tier in (3, 2, 1):
                        for rect, card in self.visible_card_hitboxes[tier]:
                            if rect.collidepoint(x, y):
                                return ("buy_card", card)
                    # kup zarezerwowaną kartę (z dolnego panelu)
                    for rect, card in self.reserved_hitboxes.get(human_player_index, []):
                        if rect.collidepoint(x, y):
                            return ("buy_card", card)
                    # wybór żetonu
                    for rect, color in self.token_hitboxes:
                        if rect.collidepoint(x, y):
                            return ("select_gem", color)

                if ev.button == 3:
                    # rezerwacja widocznej karty
                    for tier in (3, 2, 1):
                        for rect, card in self.visible_card_hitboxes[tier]:
                            if rect.collidepoint(x, y):
                                return ("reserve_visible", card)
                    # rezerwacja ze stosu
                    for tier, rect in self.deck_hitboxes.items():
                        if rect.collidepoint(x, y):
                            return ("reserve_deck", tier)
                    return ("clear_selection",)

        return None