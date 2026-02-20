from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class HIDConfig:
    enabled: bool = False
    vendor_id: Optional[int] = None
    product_id: Optional[int] = None


class HIDListener:
    def __init__(self, cfg: HIDConfig, on_trigger: Callable[[], None]) -> None:
        self.cfg = cfg
        self.on_trigger = on_trigger
        self._running = False

    def start(self) -> None:
        if not self.cfg.enabled:
            return
        try:
            import hid  # type: ignore
        except Exception:
            print("HID disabled: 'hid' package not available")
            return
        self._running = True
        import threading
        t = threading.Thread(target=self._loop, args=(), daemon=True)
        t.start()

    def _loop(self) -> None:
        try:
            import hid  # type: ignore
            dev = hid.device()
            if self.cfg.vendor_id and self.cfg.product_id:
                dev.open(self.cfg.vendor_id, self.cfg.product_id)
            else:
                print("HID: vendor/product IDs not set; skipping")
                return
            dev.set_nonblocking(True)
            print("HID listener started")
            while self._running:
                data = dev.read(64)
                if data:
                    # naive: any packet means trigger
                    self.on_trigger()
        except Exception as ex:
            print(f"HID error: {ex}")

    def stop(self) -> None:
        self._running = False
