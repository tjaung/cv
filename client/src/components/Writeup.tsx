import type { ReactNode } from 'react'

export default function Writeup({ title, children }: { title: string; children: ReactNode }) {
  return <details className="panel preprocessing-writeup" open>
    <summary>{title}</summary>
    <article className="preprocessing-writeup-content">{children}</article>
  </details>
}
