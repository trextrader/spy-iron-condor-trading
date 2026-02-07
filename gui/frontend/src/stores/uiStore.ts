import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration?: number;
}

interface UIState {
  // Sidebar
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;

  // Theme (future use)
  theme: 'dark' | 'light';
  setTheme: (theme: 'dark' | 'light') => void;

  // Notifications
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;

  // Modal state
  activeModal: string | null;
  modalData: unknown;
  openModal: (modalId: string, data?: unknown) => void;
  closeModal: () => void;

  // WebSocket connection
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set, get) => ({
        // Sidebar
        sidebarCollapsed: false,
        toggleSidebar: () => {
          set(
            (state) => ({ sidebarCollapsed: !state.sidebarCollapsed }),
            false,
            'toggleSidebar'
          );
        },

        // Theme
        theme: 'dark',
        setTheme: (theme) => {
          set({ theme }, false, 'setTheme');
        },

        // Notifications
        notifications: [],
        addNotification: (notification) => {
          const id = `notif-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
          const newNotification = { ...notification, id };

          set(
            (state) => ({
              notifications: [...state.notifications, newNotification],
            }),
            false,
            'addNotification'
          );

          // Auto-remove after duration (default 5s)
          const duration = notification.duration ?? 5000;
          if (duration > 0) {
            setTimeout(() => {
              get().removeNotification(id);
            }, duration);
          }
        },

        removeNotification: (id) => {
          set(
            (state) => ({
              notifications: state.notifications.filter((n) => n.id !== id),
            }),
            false,
            'removeNotification'
          );
        },

        clearNotifications: () => {
          set({ notifications: [] }, false, 'clearNotifications');
        },

        // Modal
        activeModal: null,
        modalData: null,
        openModal: (modalId, data = null) => {
          set({ activeModal: modalId, modalData: data }, false, 'openModal');
        },
        closeModal: () => {
          set({ activeModal: null, modalData: null }, false, 'closeModal');
        },

        // WebSocket
        wsConnected: false,
        setWsConnected: (connected) => {
          set({ wsConnected: connected }, false, 'setWsConnected');
        },
      }),
      {
        name: 'condorbrain-ui',
        partialize: (state) => ({
          sidebarCollapsed: state.sidebarCollapsed,
          theme: state.theme,
        }),
      }
    ),
    { name: 'ui-store' }
  )
);
