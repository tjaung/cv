import { useState } from 'react'
import type { PCAReport } from '../api'

type Row = PCAReport & { archived_at?: string }
export type SortState = { key: string; direction: 'asc' | 'desc' }

function value(row: Row, key: string, selected: string): number | string | null {
  const e = row.evaluation
  const image = e?.rows.find((r) => r.image_id === `metal_plate/test/${selected}`)
  if (key.startsWith('class:')) {
    const rows = e?.rows.filter((r) => r.label === key.slice(6)) ?? []
    return rows.length ? rows.filter((r) => r.prediction === r.actual).length / rows.length : null
  }
  switch (key) {
    case 'accuracy': return e?.images ? (e.tp + e.tn) / e.images : null
    case 'balanced': return e?.recall != null && e.false_positive_rate != null ? (e.recall + 1 - e.false_positive_rate) / 2 : null
    case 'performance': case 'recall': return e?.recall ?? null
    case 'precision': return e?.precision ?? null
    case 'f1': return e && 2 * e.tp + e.fp + e.fn ? 2 * e.tp / (2 * e.tp + e.fp + e.fn) : null
    case 'fpr': return e?.false_positive_rate ?? null
    case 'tp': case 'fp': case 'tn': case 'fn': return e?.images ? e[key] : null
    case 'correct': return image ? Number(image.prediction === image.actual) : null
    case 'score': return image?.score ?? null
    case 'prediction': return image?.prediction ?? null
    case 'actual': return image?.actual ?? null
    case 'margin': return image ? image.score - row.plate_threshold : null
    case 'anomalous': return image?.anomalous_patches ?? null
    case 'threshold': return row.plate_threshold
    case 'images': return e?.images ?? null
    case 'training': return row.training_images
    case 'components': return row.components
    case 'variance': return row.retained_variance
    case 'features': return `${row.config.feature_set ?? 'lab_sobel'} ${row.config.pipeline_signature ?? ''}`
    case 'parameters': return JSON.stringify(row.config)
    case 'archived': return row.archived_at ? Date.parse(row.archived_at) : Number.MAX_SAFE_INTEGER
    default: return `${row.name} ${row.model_id}`
  }
}

function compare(a: number | string | null, b: number | string | null, direction: SortState['direction']) {
  // Missing results stay last regardless of direction.
  if (a === null || b === null) return a === b ? 0 : a === null ? 1 : -1
  const order = typeof a === 'number' && typeof b === 'number' ? a - b : String(a).localeCompare(String(b), undefined, { numeric: true })
  return direction === 'asc' ? order : -order
}

export function sortModels<T extends Row>(rows: T[], sort: SortState, selected = ''): T[] {
  return [...rows].sort((a, b) => {
    let order = compare(value(a, sort.key, selected), value(b, sort.key, selected), sort.direction)
    if (!order && sort.key === 'performance') order = compare(value(a, 'fpr', selected), value(b, 'fpr', selected), sort.direction === 'desc' ? 'asc' : 'desc')
    return order || a.model_id.localeCompare(b.model_id)
  })
}

export function useModelSort<T extends Row>(rows: T[], initial: string, selected = '') {
  const [sort, setSort] = useState<SortState>({ key: initial, direction: 'desc' })
  const choose = (key: string) => setSort((current) => ({ key, direction: current.key === key ? current.direction === 'asc' ? 'desc' : 'asc' : ['name', 'parameters', 'features', 'fpr', 'fp', 'fn', 'actual', 'prediction'].includes(key) ? 'asc' : 'desc' }))
  return { sort, choose, setSort, rows: sortModels(rows, sort, selected) }
}
