# scripts/test_move_zero.py
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from src.robot.go2_interface import Go2Interface

ChannelFactoryInitialize(0, 'enx98fc84e68f1a')
go2 = Go2Interface(already_initialized=True)

print("Standing up...")
go2.stand()
time.sleep(2)

print("Watching feet for 5 seconds while standing (baseline)...")
time.sleep(5)
go2._sport.SwitchGait(1)
print("Commanding Move(0, 0, 0) for 5 seconds...")
go2.move(vx=0.1, vy=0.01, vyaw=0.0, duration=5.0)
go2.move(vx=-0.1, vy=-0.01, vyaw=0.0, duration=5.0)

print("Stopped. Observe final pose.")
input("Press Enter to sit")
go2.sit()
go2.__getstate__