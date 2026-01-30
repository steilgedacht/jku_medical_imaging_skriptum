#import "@preview/classicthesis:0.1.0": *

#set page(
  margin: 2.0cm,
)

#show par: set par(first-line-indent: 0pt)
#set par(spacing: 1.2em) 

#show heading: it => {
  set text(tracking: 0pt)
  it.body
}

#let old-definition = definition
#let definition(title: none, body) = block(
  fill: red.lighten(91%), 
  inset: 8pt,          
  radius: 2pt,          
  width: 100%,
  old-definition(title: title, body)
)
#show table: set table(
  // 1. Define the stroke (lines) - using a light gray for a clean look
  stroke: 0.5pt + gray.lighten(50%),
  
  // 2. The magic coloring function
  fill: (column, row) => {
    if row == 0 {
      // Header row color (matching your dark red theme)
      red.darken(20%)
    } else if calc.even(row) {
      // Alternating row color (very light gray or light red)
      red.lighten(95%)
    } else {
      // Odd rows stay white
      white
    }
  }
)

// 3. To make the text in the header white automatically
#show table: it => {
  show table.cell.where(y: 0): set text(fill: white, weight: "bold")
  it
}
