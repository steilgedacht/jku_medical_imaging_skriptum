#import "@preview/classicthesis:0.1.0": *

#show: classicthesis.with(
  title: "Medical Imaging",
  subtitle: "Lecture Skriptum",
  author: "Benjamin Bergmann",
  date: "2026",
)

#import "template.typ": *


#include("chapters/1_Inverse_Problem.typ")
#include("chapters/2_Xrays_and_CT.typ")
#include("chapters/3_Learned_Reconstruction.typ")
#include("chapters/4_MRI.typ")
#include("chapters/5_Image_Registration.typ")
#include("chapters/6_Segmentation.typ")
#include("chapters/7_Federated_Learning.typ")
#include("chapters/8_mircoscopy.typ")

=== A Subsection

Subsections use italic text for a subtle hierarchy.

#definition(title: "Important Concept")[
  A definition block with a distinctive left border. Use this to
  define key terms in your work.
]

#theorem(title: "Main Result")[
  A theorem block for stating important results. The numbering
  is automatic.
]

#example(title: "Practical Application")[
  An example block with a subtle gray background. Use this to
  illustrate concepts with concrete examples.
]

#remark()[
  A remark block for additional observations or notes that don't
  fit the formal structure of theorems and definitions.
]


Inline code looks like `this`, and code blocks are formatted cleanly:

```python
def hello_world():
    """A simple function."""
    print("Hello, ClassicThesis!")
```

== Tables and Figures

#figure(
  table(
    columns: (auto, auto, auto),
    table.header([*Item*], [*Description*], [*Value*]),
    [Alpha], [First item], [100],
    [Beta], [Second item], [200],
    [Gamma], [Third item], [300],
  ),
  caption: [A sample table with clean styling.],
)
