# The Computer Vision Incident or Why This Exists

Much pain and suffering has brought this repository to reality. Let this be a lesson to not just FTC teams, but developers in general

- Document everything, no matter how seemingly insignificant, no matter how irritating
  - I have a bad memory and as the second season of FTC came upon me I thought nothing of computer vision, as I had seen it all before. Yet, I had so many issues that I remember running into and solving last year, yet I didn't write any of it down so I had to go through it all again.
- Think before you delete
  - I deleted my tf25 environment that allowed training to happen which worked within the FTC SDK without thinking about why it existed. I had to go through the painful process of dependency drilling to make everything work with everything else.
- Think before you work
  - I worked on many a useless thing. Search Reddit, search the Discord, do *something* before you spend hours grinding away at something someone else already did.

## DOCUMENT IT

What happened?

I did not follow these rules, and I spent hours redoing what had already been done.

## Why does the FTC SDK require Tensorflow 2.5?

AKA why is this so painful?

From what I understand it is due to these four lines in one of the files in `org/firstinspires/ftc/robotcontroller/robotcore/external/tfod` which is really an external library jar that's hiding in the gradle cache.

```java
outputMap.put(0, outputLocations);
outputMap.put(1, outputClasses);
outputMap.put(2, outputScores);
outputMap.put(3, numDetections);
```

Doesn't this mean that you can just change this file and everything will be ok? Probably but I couldn't get the dependencies to work properly and imports to import the proper thing so good luck, and someone please fix this. There may be more to it that this. I'd like to be able to use newer version of Tensorflow. Unfortunately I don't know enough yet to fix it myself...

Why didn't you use ftcml?

Because I'm stubborn and didn't want to use cloud training credits when I had a perfectly good GPU. I'm still kind of an idiot.
