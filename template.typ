#import "@preview/classicthesis:0.1.0": *


// 2. Override margins for a uniform look
#set page(
  margin: 2.0cm,
)

// 3. Remove all paragraph indents globally
#show par: set par(first-line-indent: 0pt)
#set par(spacing: 1.2em) // Adds space between paragraphs since indents are gone

#show heading: it => {
  set text(tracking: 0pt)
  it.body
}

// Your custom definition block
#let old-definition = definition
#let definition(title: none, body) = block(
  fill: red.lighten(91%), 
  inset: 8pt,          
  radius: 2pt,          
  width: 100%,
  old-definition(title: title, body)
)
