## Setup
* Isaac Sim 4.1.0
* Omnigibson 1.1.1

## ReKep Prompt
### What will be uploaded
* Image
    * Labeled Key Points
    * Key Points Index
* Instruction
### What will be summarized
* How many stage
* Define both **sub-goal constraints** and **path constraints** as python function
    * **Input:** end-effector point and a set of keypoints
    * **Return:** numerical cost
* Keypoints to be grasped in all grasping stages by defining the `grasp_keypoints` variable
* At the end of which stage the robot should release the keypoints by defining the `release_keypoints` variable
* Don't consider collision
* Use multiple keypoint to specify a location without label (eg. mean of keypoints if location is in the center of those keypoints)

## Env
`$env:OPENAI_API_KEY="YOUR_OPENAI_API_KEY"`
