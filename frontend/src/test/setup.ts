import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'
import { setLang } from '../i18n'

// Run tests in English so queries can use English text.
setLang('en')

// jsdom does not implement these; provide stubs so components that touch them don't throw.
if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as unknown as MediaQueryList
}

if (!window.IntersectionObserver) {
  class IO {
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return []
    }
  }
  window.IntersectionObserver = IO as unknown as typeof IntersectionObserver
}

if (!window.CSS) {
  ;(window as any).CSS = { escape: (v: string) => v.replace(/["\\]/g, '\\$&') }
}

// jsdom lacks scrollIntoView (used by keyboard navigation).
Element.prototype.scrollIntoView = vi.fn()

afterEach(() => {
  cleanup()
})

