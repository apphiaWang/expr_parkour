## Expression Parkour
> Final Project for CMPT724 Affective Computing

Control parkour action with the user's  facial expression. 

| Expression | Action Unit | Parkour Action |
|-|-|-|
| Happy | AU6 Cheek Raiser, AU12 Lip Corner Puller | Jump |
| Surprised | AU2 Outer Brow Raiser, AU 27 Mouth Stretch | Slide shovel |
| Angry | AU4 Brow Lowerer, AU9 Nose Wrinkler | Dash |

### Self-evaluation
TBD

### Project Structure

- **assets**: the directory storing game assets such as background image and sound
- **data**: the directory storing our collected data for self-evaluation
    - **annotation**: a csv file for annotation
- **model**: the diectory storing code of fine-tuning transformer-based AU detection model
- **parkour.py**: the game

## How to play

### Dependencies
- Python 3.9.1
- Pygame 2.5.2
- opencv-python 4.9.0
- tensorflow 2.9.0
- DeepFace 0.0.89

After you installed the required dependencies, run `python parkour.py` to play the game.

### Authors & Contribution
|Authors | Work |
|- | - |
|Ningyi Ke | Game art design & Game model integration & Data collection |
|Yanfei Wang | Game development & game model integration & Data collection |
|Yiwen Wang | Action Unit Deection model study and fine-tuning & Data collection |

_*All authors contributed equally to the project. Authors are ordered alphabetically by name._