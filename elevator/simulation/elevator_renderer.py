"""Pygame renderer for the elevator simulation.
Responsible for: drawing the simulation, animating visual elements,
displaying statistics, displaying events, handling visual controls and input.
This module is the only one that uses pygame.
"""

import pygame
from elevator.simulation.simulation import ElevatorSimulation
from elevator.mechanics.states import ElevatorState
# LAYOUT CONSTANTS

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

HEADER_HEIGHT = 40
FOOTER_LOG_HEIGHT = 160
SIDEBAR_WIDTH = 300

BUILDING_MARGIN_LEFT = 60
BUILDING_MARGIN_TOP = 20
BUILDING_MARGIN_BOTTOM = 20

SHAFT_WIDTH = 70
SHAFT_GAP = 30
CAR_WIDTH = 54
CAR_HEIGHT_RATIO = 0.72

COLOR_BG = (18, 20, 26)
COLOR_PANEL_BG = (26, 29, 38)
COLOR_TEXT = (230, 230, 235)
COLOR_TEXT_DIM = (150, 155, 165)
COLOR_FLOOR_LINE = (70, 74, 86)
COLOR_SHAFT = (40, 43, 54)
COLOR_CAR = (86, 156, 214)
COLOR_CAR_MOVING = (110, 200, 240)
COLOR_DOOR = (230, 200, 90)
COLOR_WAITING = (240, 240, 240)
COLOR_CALL = (240, 120, 90)
COLOR_HEADER_BG = (14, 16, 21)
COLOR_LOG_BG = (14, 16, 21)
COLOR_DIVIDER = (50, 54, 66)

FONT_NAME = None


class ElevatorRenderer:
    """Renders an ElevatorSimulation to a pygame window.
    The renderer only reads simulation state;
    it never mutates position, velocity, target_floor, passenger state, or request state.
    """

    def __init__(self, simulation: ElevatorSimulation):
        self.simulation = simulation

        pygame.init()
        pygame.display.set_caption("Elevator Simulator")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock_pygame = pygame.time.Clock()

        self.font = pygame.font.SysFont(FONT_NAME, 16)
        self.font_small = pygame.font.SysFont(FONT_NAME, 13)
        self.font_bold = pygame.font.SysFont(FONT_NAME, 18, bold=True)

        self.running = True

        # Precompute building layout geometry.
        config = self.simulation.config
        self.building_area_width = WINDOW_WIDTH - SIDEBAR_WIDTH
        self.building_area_height = WINDOW_HEIGHT - HEADER_HEIGHT - FOOTER_LOG_HEIGHT
        self.floor_pixel_height = (self.building_area_height - BUILDING_MARGIN_TOP - BUILDING_MARGIN_BOTTOM) / config.num_floors
        self.pixels_per_meter = self.floor_pixel_height / config.floor_height
        self.building_bottom_px = HEADER_HEIGHT + self.building_area_height - BUILDING_MARGIN_BOTTOM

        self._speed_options = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    def run(self) -> None:
        """Run the render/update loop until the window is closed."""
        while self.running:
            real_dt = self.clock_pygame.tick(self.simulation.config.render_fps) / 1000.0
            self._handle_input()
            self.simulation.step(real_dt)
            self._render()

        pygame.quit()

    def _handle_input(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

    def _handle_key(self, key) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            self.simulation.clock.toggle_pause()
        elif key == pygame.K_r:
            self.simulation.reset()
        elif key == pygame.K_1:
            self.simulation.clock.set_time_scale(0.25)
        elif key == pygame.K_2:
            self.simulation.clock.set_time_scale(0.5)
        elif key == pygame.K_3:
            self.simulation.clock.set_time_scale(1.0)
        elif key == pygame.K_4:
            self.simulation.clock.set_time_scale(2.0)
        elif key == pygame.K_5:
            self.simulation.clock.set_time_scale(5.0)
        elif key == pygame.K_6:
            self.simulation.clock.set_time_scale(10.0)


    def _sim_y_to_screen_y(self, position_y: float) -> float:
        """Convert a simulation space vertical position to a screen y."""
        return self.building_bottom_px - position_y * self.pixels_per_meter

    def _floor_to_screen_y(self, floor_index: int) -> float:
        floor_pos = floor_index * self.simulation.config.floor_height
        return self._sim_y_to_screen_y(floor_pos)

    def _shaft_x(self, elevator_id: int) -> float:
        return BUILDING_MARGIN_LEFT + elevator_id * (SHAFT_WIDTH + SHAFT_GAP)

    def _render(self) -> None:
        self.screen.fill(COLOR_BG)
        self._render_header()
        self._render_building()
        self._render_sidebar()
        self._render_event_log()
        pygame.display.flip()

    def _render_header(self) -> None:
        pygame.draw.rect(self.screen, COLOR_HEADER_BG, (0, 0, WINDOW_WIDTH, HEADER_HEIGHT))
        title = self.font_bold.render("ELEVATOR SIMULATOR", True, COLOR_TEXT)
        self.screen.blit(title, (16, 8))

    def _render_building(self) -> None:
        config = self.simulation.config
        building = self.simulation.building

        # Floor lines and labels.
        for floor in range(config.num_floors):
            y = self._floor_to_screen_y(floor)
            pygame.draw.line(self.screen, COLOR_FLOOR_LINE,
        (BUILDING_MARGIN_LEFT - 10, y),(self.building_area_width - 10, y),1)
            label = self.font_small.render(str(floor + 1), True, COLOR_TEXT_DIM)
            self.screen.blit(label, (BUILDING_MARGIN_LEFT - 40, y - 8))

            waiting = building.waiting_at(floor)
            self._render_waiting_passengers(waiting, y)
            if waiting:
                self._render_call_indicator(y)

        # Elevator shafts + cars.
        for elevator in building.elevators:
            self._render_shaft(elevator)
            self._render_elevator_car(elevator)

    def _render_waiting_passengers(self, waiting, floor_screen_y: float) -> None:
        x = self.building_area_width - 200
        for i, passenger in enumerate(waiting[:6]):
            px = x + (i % 3) * 60
            py = floor_screen_y - 24 - (i // 3) * 20
            pygame.draw.circle(self.screen, COLOR_WAITING, (int(px), int(py)), 5)
            pygame.draw.rect(self.screen, COLOR_WAITING, (px - 4, py + 4, 8, 10))
            dest_label = self.font_small.render(f"->{passenger.destination_floor + 1}", True, COLOR_TEXT_DIM)
            self.screen.blit(dest_label, (px + 8, py - 4))

    def _render_call_indicator(self, floor_screen_y: float) -> None:
        x = self.building_area_width - 235
        pygame.draw.polygon(self.screen, COLOR_CALL,
        [(x, floor_screen_y - 6), (x + 10, floor_screen_y - 6), (x + 5, floor_screen_y - 16)])

    def _render_shaft(self, elevator) -> None:
        x = self._shaft_x(elevator.id)
        top_y = self._floor_to_screen_y(self.simulation.config.num_floors - 1)
        bottom_y = self._floor_to_screen_y(0)
        pygame.draw.rect(self.screen, COLOR_SHAFT,(x, top_y - 10, SHAFT_WIDTH, bottom_y - top_y + 10))

    def _render_elevator_car(self, elevator) -> None:
        shaft_x = self._shaft_x(elevator.id)
        car_center_x = shaft_x + SHAFT_WIDTH / 2
        car_screen_y = self._sim_y_to_screen_y(elevator.position_y)
        car_height = self.floor_pixel_height * CAR_HEIGHT_RATIO

        car_rect = pygame.Rect(car_center_x - CAR_WIDTH / 2, car_screen_y - car_height, CAR_WIDTH, car_height,)

        is_moving = elevator.state == ElevatorState.MOVING
        color = COLOR_CAR_MOVING if is_moving else COLOR_CAR
        pygame.draw.rect(self.screen, color, car_rect, border_radius=4)

        # Door animation: two panels sliding apart based on door_progress.
        self._render_doors(car_rect, elevator.door_progress)

        # Direction arrow.
        arrow = {"UP": "^", "DOWN": "v", "IDLE": "-"}[elevator.direction.name]
        label = self.font_bold.render(f"{elevator.id} {arrow}", True, (10, 10, 15))
        label_rect = label.get_rect(center=(car_rect.centerx, car_rect.top - 12))
        self.screen.blit(label, label_rect)

        # Passenger count.
        count_label = self.font_small.render(f"{len(elevator.passengers)}/{elevator.config.elevator_capacity}",True,COLOR_TEXT)
        count_rect = count_label.get_rect(center=(car_rect.centerx, car_rect.bottom + 10))
        self.screen.blit(count_label, count_rect)

    def _render_doors(self, car_rect: pygame.Rect, door_progress: float) -> None:
        gap = car_rect.width * 0.9 * door_progress
        panel_width = (car_rect.width - gap) / 2

        left_panel = pygame.Rect(car_rect.left, car_rect.top, max(0, panel_width), car_rect.height)
        right_panel = pygame.Rect(car_rect.right - max(0, panel_width), car_rect.top, max(0, panel_width), car_rect.height)
        pygame.draw.rect(self.screen, COLOR_DOOR, left_panel)
        pygame.draw.rect(self.screen, COLOR_DOOR, right_panel)

    def _render_sidebar(self) -> None:
        x0 = WINDOW_WIDTH - SIDEBAR_WIDTH
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, (x0, HEADER_HEIGHT, SIDEBAR_WIDTH, self.building_area_height))
        pygame.draw.line(self.screen, COLOR_DIVIDER, (x0, HEADER_HEIGHT), (x0, HEADER_HEIGHT + self.building_area_height), 1)

        snap = self.simulation.get_statistics_snapshot()
        clock = self.simulation.clock

        lines = [("SIMULATION", None),
            (f"Time: {snap['simulation_time']:.1f} s", COLOR_TEXT),
            (f"Speed: {clock.time_scale:g}x{' (paused)' if clock.paused else ''}", COLOR_TEXT),
            ("", None),
            (f"Waiting: {snap['waiting_passengers']}", COLOR_TEXT),
            (f"Inside: {snap['passengers_inside']}", COLOR_TEXT),
            (f"Completed: {snap['completed_passengers']}", COLOR_TEXT),
            (f"Avg wait: {snap['average_waiting_time']:.1f}s", COLOR_TEXT),
            (f"Max wait: {snap['max_waiting_time']:.1f}s", COLOR_TEXT),
            (f"Avg journey: {snap['average_journey_time']:.1f}s", COLOR_TEXT),
            ("", None),
            ("CONTROLS", None),
            ("SPACE  pause/resume", COLOR_TEXT_DIM),
            ("1-6    set speed", COLOR_TEXT_DIM),
            ("R      reset", COLOR_TEXT_DIM),
            ("ESC    quit", COLOR_TEXT_DIM)]

        y = HEADER_HEIGHT + 14
        for text, color in lines:
            if color is None:
                if text:
                    surf = self.font_bold.render(text, True, COLOR_TEXT)
                    self.screen.blit(surf, (x0 + 16, y))
                y += 22
                continue
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (x0 + 16, y))
            y += 20

    def _render_event_log(self) -> None:
        y0 = WINDOW_HEIGHT - FOOTER_LOG_HEIGHT
        pygame.draw.rect(self.screen, COLOR_LOG_BG, (0, y0, WINDOW_WIDTH, FOOTER_LOG_HEIGHT))
        pygame.draw.line(self.screen, COLOR_DIVIDER, (0, y0), (WINDOW_WIDTH, y0), 1)

        header = self.font_bold.render("EVENT LOG", True, COLOR_TEXT)
        self.screen.blit(header, (16, y0 + 8))

        max_visible = (FOOTER_LOG_HEIGHT - 36) // 18
        recent = self.simulation.events.recent(max_visible)

        y = y0 + 34
        for evt in recent:
            line = f"{evt.simulation_time:6.1f}  {evt.message}"
            surf = self.font_small.render(line, True, COLOR_TEXT_DIM)
            self.screen.blit(surf, (16, y))
            y += 18
