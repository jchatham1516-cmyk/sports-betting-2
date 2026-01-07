import SwiftUI

struct RootTabView: View {
    var body: some View {
        TabView {
            RunView()
                .tabItem {
                    Label("Run", systemImage: "play.circle")
                }
            BetsView()
                .tabItem {
                    Label("Bets", systemImage: "list.bullet")
                }
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
        }
    }
}

#Preview {
    RootTabView()
}
