# Dataset Structure

### Denver Intensity of Spontaneous Facial Action (DISFA) Database 

DISFA database is a non-posed facial expression database, which contains the collection spontaneous facial actions of 27 adult subjects, and each subject has been video recorded for 4 minutes to provide 130,788 images in total. 

We used DISFA to train our “transformer based AU model”. We did not post this database, since it is only available for research purpose. If you want to use it to run our code, you need to complete the DISFA agreement form (http://mohammadmahoor.com/disfa-contact-form/) to request DISFA data.


## <ins>Self Created Evaluation Dataset</ins>

### Expression Video
This is a dataset containing video of three Eastern-Asian female authors' facial expression while playing the project game. There are 23 videos ranging from 40 to 60 seconds long, with a total of 303 facial expressions. This dataset was captured by a BISON CAM, NB Pro webcam. The video is named as form "A_B", where A is the experimenter's number and B is the video's number. The naming matches the information in the Annotation.csv file.


### Annotation.csv
This CSV file is constructed by project members who manually annotated the model's prediction of the expression in each frame, as well as their own defined ground truth, and corresponding time points based on the facial expressions in the 'Expression Video' mentioned above.

#### Content structure:
* Video: The unique identifier of a video, which records the encoding of the source video.
* Start_time / end_time: respectively represent the start and end times of recording emotional states, represented by timestamps in the video.
* Label: Each label represents the subjective judgment and classification of video content by team members.
* Pred: The prediction results of expression recognition models for the same context, used for comparison with artificial labels.


### Disclaimer: Please do not use or publish the dataset on other platforms without permission.

