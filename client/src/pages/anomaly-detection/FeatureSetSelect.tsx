export default function FeatureSetSelect({ value, disabled, onChange }: { value: string; disabled: boolean; onChange: (value: string) => void }) {
  return <label>Features <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
    <option value="lab_sobel">LAB + Sobel (105)</option>
    <option value="lab_hog">LAB + HOG (366)</option>
    <option value="lab_sobel_hog">LAB + Sobel + HOG (429)</option>
  </select></label>
}
