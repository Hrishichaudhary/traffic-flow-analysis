# Traffic Flow Analysis

## Overview
This project performs traffic flow analysis using Python scripts. It includes detailed setup and execution instructions to help users get started quickly.

## Repository Contents
- `main.py`: The main Python script for traffic flow analysis, located at the root directory.
- `Demo.mp4`: A demo video showcasing the project’s features and usage.
- `requirements.txt`: Lists the Python dependencies needed.
- `.gitignore`: Specifies files and folders excluded from the repository.

## Setup Instructions

### Prerequisites
- Python 3.x installed on your system.
- Git for cloning the repository.

### Installation Steps

1. **Clone the repository:**
* git clone https://github.com/Hrishichaudhary/traffic-flow-analysis.git
* cd traffic-flow-analysis

  
2. **Create and activate a virtual environment:**
- On Windows:
  ```
  python -m venv fresh_env
  fresh_env\Scripts\activate
  ```

3. **Install the required Python packages:**
* pip install -r requirements.txt

  
## Execution Instructions

Run the main Python script to start the traffic flow analysis: **python main.py**


## Notes on Repository Organization and Cleanup

- Large or unnecessary files and folders were intentionally excluded or removed to keep the repository clean and lightweight. This includes:
  - Directory `sort/` (previously a git submodule)
  - Output data files in `output/` folder
  - Large model files: `yolov5s.pt` and `yolov5su.pt`
- These files and folders are excluded via `.gitignore` with entries such as:
* fresh_env/
* ByteTrack/
* sort/
* output/
* yolov5s.pt
* yolov5su.pt
* .mp4
* .dll
* .lib
* pycache/.pyc
* .env

- The demo video file `Demo.mp4` is included deliberately by force-adding it to the repository.

## Demo Video

The included `Demo.mp4` file visually demonstrates the project setup and basic execution.

## Contributing

Feel free to open issues or pull requests for improvements.


