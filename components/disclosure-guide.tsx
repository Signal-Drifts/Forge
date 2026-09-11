'use client';

import { useId, useState } from 'react';
import { ChevronRight } from 'lucide-react';

export function DisclosureGuide({
  title,
  countLabel,
  children,
}: {
  title: string;
  countLabel: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const panelId = useId();

  return (
    <section className="secondary-disclosure">
      <button
        className="secondary-disclosure-trigger"
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
      >
        <ChevronRight aria-hidden="true" />
        <strong>{title}</strong>
        <span>{countLabel}</span>
      </button>
      <div id={panelId} className="secondary-disclosure-content" hidden={!open}>
        {children}
      </div>
    </section>
  );
}
