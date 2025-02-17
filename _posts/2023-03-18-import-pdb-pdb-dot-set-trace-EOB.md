---
title: "import pdb; pdb.set_trace() <EOB>"
date: '2023-03-18'
layout: post
categories: Proteomics
---

Take my word for this. This will help you for years to come if you are in data science.
By the way,
```python
import pdb; pdb.set_trace()
```

Just include it anywhere in your python script for debugging.  It will put the script in the debug mode. In the debug mode, other than the general features of viewing the current variables etc, I always add new variables, come up with new functionality that I can later implement in the original script after quitting the debug mode.  

It helped me a ton in the last couple of scripts I wrote, and I thought **it** deserves a blog post by itself. My original intention is to just include the command in the title, and NOT include any body in the post. However I ended up writing few lines.  So, when you use this next time, do NOT include **_\<EOB\>_** in the debug mode :-), just do 

```python
import pdb; pdb.set_trace()
```


More info here: [https://docs.python.org/2/library/pdb.html](https://docs.python.org/2/library/pdb.html)
