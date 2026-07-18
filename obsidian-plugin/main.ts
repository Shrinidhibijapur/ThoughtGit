import { App, Plugin, PluginSettingTab, Setting, TFile, Notice, requestUrl } from 'obsidian';

interface ThoughtGitSettings {
    serverUrl: string;
    enableAutoSync: boolean;
    activeBranch: string;
}

const DEFAULT_SETTINGS: ThoughtGitSettings = {
    serverUrl: 'http://localhost:8765',
    enableAutoSync: true,
    activeBranch: 'main'
};

export default class ThoughtGitPlugin extends Plugin {
    settings: ThoughtGitSettings = DEFAULT_SETTINGS;
    private debounceTimer: number | null = null;

    async onload() {
        console.log('Loading ThoughtGit Plugin...');
        await this.loadSettings();

        // 1. Settings Tab
        this.addSettingTab(new ThoughtGitSettingTab(this.app, this));

        // 2. Vault Modification Listeners
        this.registerEvent(
            this.app.vault.on('modify', (file) => {
                if (this.settings.enableAutoSync && file instanceof TFile && file.extension === 'md') {
                    this.triggerSync(file);
                }
            })
        );

        // 3. Ribbon Icon Indicator
        this.addRibbonIcon('brain', 'ThoughtGit Status', () => {
            new Notice(`ThoughtGit is Active\nServer: ${this.settings.serverUrl}\nBranch: ${this.settings.activeBranch}`);
        });

        // 4. Command: Bulk Index Vault
        this.addCommand({
            id: 'thoughtgit-bulk-index',
            name: 'Bulk Index Vault Memory',
            callback: async () => {
                await this.bulkIndexVault();
            }
        });
    }

    onunload() {
        console.log('Unloading ThoughtGit Plugin...');
    }

    private triggerSync(file: TFile) {
        // Debounce to avoid saving on every single keystroke
        if (this.debounceTimer) {
            window.clearTimeout(this.debounceTimer);
        }

        this.debounceTimer = window.setTimeout(async () => {
            try {
                const text = await this.app.vault.read(file);
                if (text.trim().length > 10) {
                    await this.syncThought(text, file.name);
                }
            } catch (err) {
                console.error('ThoughtGit: Read failed', err);
            }
        }, 2500); // 2.5-second debounce window
    }

    private async syncThought(content: string, fileName: string) {
        const url = `${this.settings.serverUrl}/ingest`;
        const payload = {
            content: content,
            source: 'obsidian',
            metadata: {
                file_name: fileName,
                branch: this.settings.activeBranch
            }
        };

        try {
            const response = await requestUrl({
                url: url,
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (response.status === 200) {
                console.log(`ThoughtGit: Synced file "${fileName}" successfully.`);
            } else {
                console.warn(`ThoughtGit: Server returned status ${response.status}`);
            }
        } catch (err) {
            console.error('ThoughtGit: Connection to FastAPI server failed:', err);
        }
    }

    private async bulkIndexVault() {
        const files = this.app.vault.getMarkdownFiles();
        if (files.length === 0) {
            new Notice('No markdown files found in the vault to index.');
            return;
        }

        new Notice(`ThoughtGit: Ingesting ${files.length} markdown notes...`);
        let successCount = 0;

        for (const file of files) {
            try {
                const text = await this.app.vault.read(file);
                if (text.trim().length > 10) {
                    await this.syncThought(text, file.name);
                    successCount++;
                }
            } catch (e) {
                console.error(`ThoughtGit: Failed to read file ${file.name}`, e);
            }
        }

        new Notice(`ThoughtGit: Successfully indexed ${successCount} notes!`);
    }

    async loadSettings() {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    }

    async saveSettings() {
        await this.saveData(this.settings);
    }
}

class ThoughtGitSettingTab extends PluginSettingTab {
    plugin: ThoughtGitPlugin;

    constructor(app: App, plugin: ThoughtGitPlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        const { containerEl } = this;
        containerEl.empty();
        containerEl.createEl('h2', { text: 'ThoughtGit Memory Sync Settings' });

        // Server URL input setting
        new Setting(containerEl)
            .setName('Server Backend URL')
            .setDesc('Host URL where your ThoughtGit FastAPI backend server is running.')
            .addText(text => text
                .setPlaceholder('http://localhost:8765')
                .setValue(this.plugin.settings.serverUrl)
                .onChange(async (value) => {
                    this.plugin.settings.serverUrl = value.trim();
                    await this.plugin.saveSettings();
                }));

        // Enable/Disable auto background syncing toggle
        new Setting(containerEl)
            .setName('Enable Background Auto-Sync')
            .setDesc('Silently indexes notes in the background as you edit them.')
            .addToggle(toggle => toggle
                .setValue(this.plugin.settings.enableAutoSync)
                .onChange(async (value) => {
                    this.plugin.settings.enableAutoSync = value;
                    await this.plugin.saveSettings();
                }));

        // Branch targeting setting
        new Setting(containerEl)
            .setName('Target Sync Branch')
            .setDesc('The repository branch namespace to index notes into.')
            .addText(text => text
                .setPlaceholder('main')
                .setValue(this.plugin.settings.activeBranch)
                .onChange(async (value) => {
                    this.plugin.settings.activeBranch = value.trim().toLowerCase() || 'main';
                    await this.plugin.saveSettings();
                }));
    }
}
