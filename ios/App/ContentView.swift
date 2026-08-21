import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            LiveView()
                .tabItem { Label("Live", systemImage: "camera.viewfinder") }
            BenchView()
                .tabItem { Label("Bench", systemImage: "speedometer") }
        }
    }
}
