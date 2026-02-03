#import "@preview/classicthesis:0.1.0": *
#show table: set table(
  // 1. Define the stroke (lines) - using a light gray for a clean look
  stroke: 0.5pt + gray.lighten(50%),
  
  // 2. The magic coloring function
  fill: (column, row) => {
    if row == 0 {
      // Header row color (matching your dark red theme)
      green.darken(20%)
    } else if calc.even(row) {
      // Alternating row color (very light gray or light red)
      green.lighten(95%)
    } else {
      // Odd rows stay white
      white
    }
  }
)

#show table: it => {
  show table.cell.where(y: 0): set text(fill: white, weight: "bold")
  it
}
#show par: set par(first-line-indent: 0pt)

#show: classicthesis.with(
  title: "Medical Imaging",
  subtitle: "Lecture Skriptum",
  author: "Benjamin Bergmann",
  date: "2026",
)

#set page(
  margin: 2.0cm,
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

