# patternlab
We are attempting to reproduce the research paper (Link 1) as part of our Pattern Recognition (Lab) course.
This repo will have all the codes and track the progress for our course.

# Demo
To have a demo of the current progress in a more presentable way, check out the Project_demo.ipynb file.

# Links
These are the links provided in the class and will be used all throughout the course.

1. Implementing word2vec: https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb

## Work Distribution
- [x] Grasp the ideas written in the 1st link. The article needs to be read by everyone and in case of difficulty, contact @rifazn
- [x] Implement the code from the 1st link as much as possible by ourselves in a manner that maximises our understanding of the topics (by doing research in between, not just code)
    - [x] Find the right probabilities of every center,context pair (in form of P(context|center))
    - [x] Make word2idx and idx2word arrays, so they can be passed to 'torch'
    - [x] Finish rest of the code (the PyTorch parts)
- [x] Finally, run the code, and do an excellent demo of what the code can do in class. Have presentation slides ready.
- [ ] The blog's code only calculates the loss of the predictions, doesn't actually show the suggested context given center. We might need to do that
  extra part.
    
