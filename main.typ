#import "@preview/classicthesis:0.1.0": *

// 1. Set the global paragraph override BEFORE the template call
#show par: set par(first-line-indent: 0pt)

// 2. Configure the template with twoside: false
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

