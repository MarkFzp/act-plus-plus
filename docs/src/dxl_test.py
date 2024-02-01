from dynamixel_client import DynamixelClient
client = DynamixelClient([1, 2], port='/dev/ttyDXL_wheels', lazy_connect=True)

print(client.read_pos_vel_cur())
