import * as React from 'react'

/**
 * Detects whether the user is actively scrolling within a container element.
 * Returns `true` while the scroll is in progress and for `idleMs` after it stops,
 * then returns `false`.
 *
 * @param containerRef - A ref to the scrollable container element.
 * @param idleMs - How long after the last scroll event to consider the scroll "idle" (default 1500ms).
 */
export function useScrollActivity(
  containerRef: React.RefObject<HTMLElement | null>,
  idleMs = 1500,
): boolean {
  const [isScrolling, setIsScrolling] = React.useState(false)
  const timerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  React.useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const handleScroll = () => {
      setIsScrolling(true)

      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }

      timerRef.current = setTimeout(() => {
        setIsScrolling(false)
        timerRef.current = null
      }, idleMs)
    }

    el.addEventListener('scroll', handleScroll, { passive: true })

    return () => {
      el.removeEventListener('scroll', handleScroll)
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
    }
  }, [containerRef, idleMs])

  return isScrolling
}
