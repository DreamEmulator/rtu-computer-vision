# 🔍 Images for analyzing

Step 0: a photo of what your product should recognise. Every pipeline run sends it through every stage, and every stage shows you what it did to it: in its report, in its `before_after.png`, and on the Demo Day page.

- **One image per run:** the last one by name, if there are several. A photo URL pasted into **Actions → 🚀 CI-Pipeline → Run workflow** wins.
- **Commit a new one** and CI runs by itself.
- **Not training data, and not the test set.** The pipeline never learns from these images and never grades itself on them. It learns from [`images_for_training/`](../images_for_training/), and stage 1 locks away part of those as the test set.
