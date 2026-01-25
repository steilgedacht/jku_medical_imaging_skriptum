#import "@preview/classicthesis:0.1.0": *

#show heading: it => {
  set text(tracking: 0pt)
  it.body
}
#set page(
  margin: (inside: 2.0cm, outside: 2.0cm), // Gleiche Werte für innen und außen
)

#set par(first-line-indent: 0pt)

#let old-definition = definition

#let definition(title: none, body) = block(
  fill: red.lighten(91%), // Sehr helles Rot
  stroke: red.lighten(80%) + 0.5pt, // Optional: feiner roter Rand
  inset: 8pt,             // Abstand vom Text zum Rand der Box
  radius: 2pt,            // Leicht abgerundete Ecken
  width: 100%,
  old-definition(title: title, body)
)


