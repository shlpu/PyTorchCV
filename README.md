### Pose Estimation

---
##### Command to Train COCO Keypoints:
```Python
python main.py --dataset coco --hypes ./hypes/coco_pose.json --phase train
```

##### Command to Test COCO Keypoints:
```Python
python main.py --dataset coco --hypes ./hypes/coco_pose.json --phase test
```

---
##### Command to Train Laneline Pose:
```Python
python main.py --dataset lane --hypes ./hypes/coco_pose.json --phase train
```

##### Command to Test Laneline Pose:
```Python
python main.py --dataset lane --hypes ./hypes/coco_pose.json --phase test
```