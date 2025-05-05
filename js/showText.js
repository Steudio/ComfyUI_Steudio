import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "Steudio.UI",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (
            nodeData.name === "Ratio Calculator" ||
            nodeData.name === "Sequence Generator" ||
            nodeData.name === "Display UI"
        ) {
            function populate(text) {
                // Clear existing widgets
                if (this.widgets) {
                    const isConvertedWidget = +!!this.inputs?.[0].widget;
                    for (let i = isConvertedWidget; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = isConvertedWidget;
                }

                // Create widget to display the text
                const widget = ComfyWidgets["STRING"](
                    this,
                    "text_box",
                    ["STRING", { multiline: true }],
                    app
                ).widget;

                widget.inputEl.readOnly = true;
                widget.inputEl.style.opacity = 0.6;

                // Apply double return-to-line only for "Display UI"
                widget.value =
                    nodeData.name === "Display UI"
                        ? Array.isArray(text) ? text.map(line => line + "\n").join("\n") : text // Ensures explicit line breaks
                        : Array.isArray(text) ? text.join("") : text;
            }

            // Update node execution logic
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                populate.call(this, message.text);
            };

            const VALUES = Symbol();
            const configure = nodeType.prototype.configure;
            nodeType.prototype.configure = function () {
                this[VALUES] = arguments[0]?.widgets_values;
                return configure?.apply(this, arguments);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                const widgets_values = this[VALUES];

                if (widgets_values?.length) {
                    requestAnimationFrame(() => {
                        populate.call(this, widgets_values[0]);
                    });
                }
            };
        }
    },
});
