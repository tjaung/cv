import type { SortState } from '../hooks/useModelSort'

export default function SortHeader({ column, children, sort, onSort }: { column: string; children: React.ReactNode; sort: SortState; onSort: (column: string) => void }) {
  const active = sort.key === column
  return <th scope="col" aria-sort={active ? sort.direction === 'asc' ? 'ascending' : 'descending' : 'none'}><button type="button" onClick={() => onSort(column)} title={`Sort by ${typeof children === 'string' ? children : column}`}>
    {children} {active ? sort.direction === 'asc' ? '↑' : '↓' : '↕'}
  </button></th>
}
