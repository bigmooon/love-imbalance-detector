// frontend/src/components/report/Section.tsx
import type { CSSProperties, ReactNode } from 'react'
import styles from './Section.module.scss'

interface Props {
  no: string        // "01"
  title: string     // "대화량"
  staggerIndex: number
  children: ReactNode
}

export function Section({ no, title, staggerIndex, children }: Props) {
  return (
    <section className={styles.section} style={{ '--stagger-i': staggerIndex } as CSSProperties}>
      <header className={styles.header}>
        <span className={styles.no}>{no}</span>
        <h2 className={styles.title}>{title}</h2>
      </header>
      {children}
    </section>
  )
}
