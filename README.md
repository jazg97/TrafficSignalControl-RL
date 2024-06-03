# Traffic Signal Control through Deep Reinforcement Learning

This project aims to optimize traffic signal control using deep reinforcement learning methods. It leverages the SUMO (Simulation of Urban MObility) traffic simulator with the TraCI API to create a realistic traffic environment. The project explores the effectiveness of two reinforcement learning algorithms: Rainbow DQN and PPO.

## Environment

- **SUMO (Simulation of Urban MObility)**: A traffic simulation tool that models the movement of vehicles in an urban setting.
- **TraCI API**: An interface that allows interaction between the SUMO simulator and external applications.

## Methods

- **Rainbow DQN**: A combination of several improvements to the basic DQN algorithm, including double Q-learning, prioritized replay, dueling networks, multi-step bootstrap targets, distributional DQNs, and noisy networks.
- **PPO (Proximal Policy Optimization)**: A policy gradient method that alternates between sampling data through interaction with the environment and optimizing a surrogate objective function using stochastic gradient descent.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/traffic-signal-control-drl.git
    cd traffic-signal-control-drl
    ```

2. **Set up the environment**:
    - Install [SUMO](https://www.eclipse.org/sumo/) and ensure it is added to your system's PATH.
    - Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Course Information

This project was developed as part of the "Aprendiçado por Reforço" course at UNICAMP.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the developers of SUMO and the TraCI API.
- Inspired by various research papers on traffic signal control and reinforcement learning.

## Contact

For any inquiries, please contact me at j272291@dac.unicamp.br.
