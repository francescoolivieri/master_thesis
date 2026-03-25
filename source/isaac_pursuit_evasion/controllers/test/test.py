import torch
from source.isaac_pursuit_evasion.controllers.flight_controller import betaflight_rate_profile

if __name__ == "__main__":
    # Your parameters
    rc_input = torch.tensor([1.0, 1.0, 1.0])
    rc_rate = torch.tensor([1.55, 1.55, 1.0])
    super_rate = torch.tensor([0.73, 0.73, 0.73])
    rc_expo = torch.tensor([0.3, 0.3, 0.3])

    w_cmd = betaflight_rate_profile(rc_input, rc_rate, super_rate, rc_expo, super_expo_active=True)
    print("Desired angular velocity (deg/s):", w_cmd)

    import matplotlib.pyplot as plt

    rc_rate = torch.tensor([1.58, 1.55, 1.0])
    super_rate = torch.tensor([0.73, 0.73, 0.73])
    rc_expo = torch.tensor([0.3, 0.3, 0.3])

    # Create a [100, 3] tensor with same values for each axis
    rc_input_range = torch.linspace(-1, 1, 100)
    rc_input = torch.stack([rc_input_range] * 3, dim=1)  # shape [100, 3]

    w_cmds = betaflight_rate_profile(rc_input, rc_rate, super_rate, rc_expo)  # [100, 3]

    axes = ["Roll", "Pitch", "Yaw"]
    plt.figure()
    for i in range(3):
        plt.plot(rc_input_range.numpy(), w_cmds[:, i].numpy(), label=axes[i])

    plt.title("Betaflight Rate Profile")
    plt.xlabel("RC Input")
    plt.ylabel("Desired Angular Velocity (deg/s)")
    plt.legend()
    plt.grid()
    plt.show()